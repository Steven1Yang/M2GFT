"""SelectionConv operator adapted from FastSplatStyler and Interpolated SelectionConv."""

from typing import Union
from torch_geometric.typing import OptPairTensor, Adj

import os
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch.nn.parameter import Parameter
from math import sqrt
import numpy as np

def intersect1d(tensor1,tensor2):

    device = tensor1.device

    result, ind1, ind2 = np.intersect1d(tensor1.cpu().numpy(),tensor2.cpu().numpy(),return_indices=True)

    return torch.tensor(result).to(device), torch.tensor(ind1).to(device), torch.tensor(ind2).to(device)

def setdiff1d(tensor1,tensor2):

    device = tensor1.device

    result = np.setdiff1d(tensor1.cpu().numpy(),tensor2.cpu().numpy())

    return torch.tensor(result).to(device)

class SelectionConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation = 1, padding_mode = 'zeros', **kwargs):
        super(SelectionConv, self).__init__(**kwargs)

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.selection_count = kernel_size * kernel_size
        self.matmul_chunk_size = int(os.environ.get("SELECTION_CONV_CHUNK_SIZE", "0"))
        #self.has_self_loops = has_self_loops

        self.weight = Parameter(torch.randn(self.selection_count,in_channels,out_channels,dtype=torch.float))
        torch.nn.init.uniform_(self.weight, a=-0.1, b=0.1)
        #torch.nn.init.normal_(self.weight)

        self.bias = Parameter(torch.randn(out_channels,dtype=torch.float))
        torch.nn.init.uniform_(self.bias, a=0.0, b=0.1)


    def _index_add_matmul(self, out: Tensor, x: Tensor, source: Tensor, target: Tensor, weight: Tensor, interps: Tensor = None):
        if self.matmul_chunk_size <= 0 or target.shape[0] <= self.matmul_chunk_size:
            result = torch.matmul(x[target], weight)
            if interps is not None:
                result = interps * result
            out.index_add_(0, source, result)
            return
        chunk_size = int(self.matmul_chunk_size)
        for start in range(0, target.shape[0], chunk_size):
            end = min(start + chunk_size, target.shape[0])
            result = torch.matmul(x[target[start:end]], weight)
            if interps is not None:
                result = interps[start:end] * result
            out.index_add_(0, source[start:end], result)


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, selections: Tensor, interps = None) -> Tensor:
        """"""
        
        all_nodes = torch.arange(x.shape[0]).to(x.device)

        if self.padding_mode == 'constant':
            # Constant value of the average of the all the nodes
            x_mean = torch.mean(x,dim=0)

        out = torch.zeros((x.shape[0],self.out_channels)).to(x.device)

        if self.padding_mode == 'normalize':
            dir_count = torch.zeros((x.shape[0],1)).to(x.device)

        if self.kernel_size == 1 or self.kernel_size == 3:

            # Find the appropriate node for each selection by stepping through connecting edges
            for s in range(self.selection_count):
                cur_dir = torch.where(selections == s)[0]
                
                cur_source = edge_index[0,cur_dir]
                cur_target = edge_index[1,cur_dir]
                
                if interps is not None:
                    cur_interps = interps[cur_dir]
                    cur_interps = torch.unsqueeze(cur_interps,dim=1)
                    #print(torch.amin(cur_interps),torch.amax(cur_interps))
                
                if self.dilation > 1:
                    for _ in range(1, self.dilation):
                        vals, ind1, ind2 = intersect1d(cur_target,edge_index[0,cur_dir])
                        cur_source = cur_source[ind1]
                        cur_target = edge_index[1,cur_dir][ind2]
                        if interps is not None:
                            cur_interps = cur_interps[ind1]
                    
                # Main Calculation
                self._index_add_matmul(
                    out,
                    x,
                    cur_source,
                    cur_target,
                    self.weight[s],
                    cur_interps if interps is not None else None,
                )
                
                # Sanity check
                #from tqdm import tqdm
                #for i,node in enumerate(tqdm(cur_source)):
                #    out[node] += result[i]

                if self.padding_mode == 'constant':
                    missed_nodes = setdiff1d(all_nodes, cur_source)
                    #out[missed_nodes] += torch.matmul(x_mean, self.weight[s])
                    out.index_add_(0,missed_nodes,torch.matmul(x_mean, self.weight[s]))

                if self.padding_mode == 'replicate':
                    missed_nodes = setdiff1d(all_nodes, cur_source)
                    #out[missed_nodes] += torch.matmul(x[missed_nodes], self.weight[s])
                    self._index_add_matmul(out, x, missed_nodes, missed_nodes, self.weight[s])

                if self.padding_mode == 'reflect':
                    missed_nodes = setdiff1d(all_nodes, cur_source)

                    opposite = s+4
                    if opposite > 8:
                        opposite = opposite % 9 + 1

                    op_dir = torch.where(selections == opposite)[0]

                    op_source = edge_index[0,op_dir]
                    op_target = edge_index[1,op_dir]

                    if interps is not None:
                        op_interps = interps[op_dir]
                        op_interps = torch.unsqueeze(op_interps,dim=1)
                                        
                    
                    # Only take edges that are part of missed nodes
                    vals, ind1, ind2 = intersect1d(op_source,missed_nodes)
                    op_source = op_source[ind1]
                    op_target = op_target[ind1]
                    if interps is not None:
                        op_interps = op_interps[ind1]

                    if self.dilation > 1:
                        for _ in range(1, self.dilation):
                            vals, ind1, ind2 = intersect1d(op_target,edge_index[0,op_dir])
                            op_source = op_source[ind1]
                            op_target = edge_index[1,op_dir][ind2]
                            if interps is not None:
                                op_interps = op_interps[ind1]
                    
                    # Main Calculation
                    self._index_add_matmul(
                        out,
                        x,
                        op_source,
                        op_target,
                        self.weight[s],
                        op_interps if interps is not None else None,
                    )

                if self.padding_mode == 'normalize':
                    dir_count[torch.unique(cur_source)] += 1

        else:
            width = self.kernel_size//2
            horiz = torch.arange(-width,width+1).to(x.device)
            vert = torch.arange(-width,width+1).to(x.device)

            right = torch.where(selections == 1)[0]
            left = torch.where(selections == 5)[0]
            down = torch.where(selections == 7)[0]
            up = torch.where(selections == 3)[0]

            center = torch.where(selections == 0)[0]

            # Find the appropriate node for each selection by stepping through connecting edges
            s = 0
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    x_loc = horiz[j]
                    y_loc = vert[i]

                    cur_source = edge_index[0,center] #Starting location
                    cur_target = edge_index[1,center]

                    if interps is not None:
                        cur_interps = interps[center]
                        cur_interps = torch.unsqueeze(cur_interps,dim=1)
                    
                    #print(torch.sum(cur_target-cur_source))

                    #print(cur_target.shape)

                    if x_loc < 0:
                        for _ in range(self.dilation*abs(x_loc)):
                            vals, ind1, ind2 = intersect1d(cur_target,edge_index[0,left])
                            cur_source = cur_source[ind1]
                            cur_target = edge_index[1,left][ind2]
                            if interps is not None:
                                cur_interps = cur_interps[ind1]
                    if x_loc > 0:
                        for _ in range(self.dilation*abs(x_loc)):
                            vals, ind1, ind2 = intersect1d(cur_target,edge_index[0,right])
                            cur_source = cur_source[ind1]
                            cur_target = edge_index[1,right][ind2]
                            if interps is not None:
                                cur_interps = cur_interps[ind1]

                    if y_loc < 0:
                        for _ in range(self.dilation*abs(y_loc)):
                            vals, ind1, ind2 = intersect1d(cur_target,edge_index[0,up])
                            cur_source = cur_source[ind1]
                            cur_target = edge_index[1,up][ind2]
                            if interps is not None:
                                cur_interps = cur_interps[ind1]

                    if y_loc > 0:
                        for _ in range(self.dilation*abs(y_loc)):
                            vals, ind1, ind2 = intersect1d(cur_target,edge_index[0,down])
                            cur_source = cur_source[ind1]
                            cur_target = edge_index[1,down][ind2]
                            if interps is not None:
                                cur_interps = cur_interps[ind1]

                    # Main Calculation
                    self._index_add_matmul(
                        out,
                        x,
                        cur_source,
                        cur_target,
                        self.weight[s],
                        cur_interps if interps is not None else None,
                    )
                    
                    if self.padding_mode == 'constant':
                        missed_nodes = setdiff1d(all_nodes, cur_source)
                        #out[missed_nodes] += torch.matmul(x_mean, self.weight[s])
                        out.index_add_(0,missed_nodes,torch.matmul(x_mean, self.weight[s]))

                    if self.padding_mode == 'replicate':
                        missed_nodes = setdiff1d(all_nodes, cur_source)
                        #out[missed_nodes] += torch.matmul(x[missed_nodes], self.weight[s])
                        self._index_add_matmul(out, x, missed_nodes, missed_nodes, self.weight[s])

                    if self.padding_mode == 'reflect':
                        raise ValueError("Reflect padding not yet implemented for larger kernels")

                    if self.padding_mode == 'normalize':
                        dir_count[torch.unique(cur_source)] += 1

                    s+=1

        #print(self.selection_count/(dir_count + 1e-8))
        #test_val = self.selection_count/(dir_count + 1e-8)
        # print(torch.max(test_val),torch.min(test_val),torch.mean(test_val))

        if self.padding_mode == 'zeros':
            pass  # Already accounted for in the graph structure, no further computation needed
        elif self.padding_mode == 'normalize':
            out *= self.selection_count/(dir_count + 1e-8)
        elif self.padding_mode == 'constant':
            pass # Processed earlier
        elif self.padding_mode == 'replicate':
            pass
        elif self.padding_mode == 'reflect':
            pass
        elif self.padding_mode == 'circular':
            raise ValueError("Circular padding cannot be generalized on a graph. Instead, create a graph with edges connecting to the wrapped around nodes")
        else:
            raise ValueError(f"Unknown padding mode: {self.padding_mode}")

        # Add bias if applicable
        out += self.bias

        return out


    def copy_weightsNxN(self,weight,bias=None):

        width = int(sqrt(self.selection_count))

        # Assumes weight comes in as [output channels, input channels, row, col]
        for i in range(self.selection_count):
            self.weight[i] = weight[:,:,i//width,i%width].permute(1,0)


    def copy_weights3x3(self,weight,bias=None):


        # Assumes weight comes in as [output channels, input channels, row, col]
        # Assumes weight is a 3x3

        # Current Ordering
        # 4  3  2
        # 5  0  1
        # 6  7  8

        # Need to flip horizontally per implementation of convolution
        #self.weight[5] = weight[:,:,1,2].permute(1,0)
        #self.weight[7] = weight[:,:,0,1].permute(1,0)
        #self.weight[1] = weight[:,:,1,0].permute(1,0)
        #self.weight[3] = weight[:,:,2,1].permute(1,0)
        #self.weight[6] = weight[:,:,0,2].permute(1,0)
        #self.weight[8] = weight[:,:,0,0].permute(1,0)
        #self.weight[2] = weight[:,:,2,0].permute(1,0)
        #self.weight[4] = weight[:,:,2,2].permute(1,0)
        #self.weight[0] = weight[:,:,1,1].permute(1,0)

        self.weight[1] = weight[:,:,1,2].permute(1,0)
        self.weight[3] = weight[:,:,0,1].permute(1,0)
        self.weight[5] = weight[:,:,1,0].permute(1,0)
        self.weight[7] = weight[:,:,2,1].permute(1,0)
        self.weight[2] = weight[:,:,0,2].permute(1,0)
        self.weight[4] = weight[:,:,0,0].permute(1,0)
        self.weight[6] = weight[:,:,2,0].permute(1,0)
        self.weight[8] = weight[:,:,2,2].permute(1,0)
        self.weight[0] = weight[:,:,1,1].permute(1,0)


    def copy_weights1x1(self, weight, bias=None):
        self.weight[0] = weight[:,:,0,0].permute(1, 0)


    def copy_weights(self,weight,bias=None):

        if self.kernel_size == 3:
            self.copy_weights3x3(weight,bias)
        elif self.kernel_size == 1:
            self.copy_weights1x1(weight, bias)
        else:
            self.copy_weightsNxN(weight,bias)

        if bias is None:
            self.bias[:] = 0.0
        else:
            self.bias = bias


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
