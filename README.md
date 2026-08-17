# M2GFT

[中文说明](README_CN.md)

M2GFT is a graph-based style-transfer model for 3D Gaussian splats. It is built on
the graph representation, SelectionConv operators, and R41 style-transfer path of
[FastSplatStyler (FSS)](https://github.com/davidmhart/FastSplatStyler).

FSS performs style transformation mainly at the coarse R41 graph level. M2GFT extends
this path into a four-level `r41/r31/r21/r11` graph feature pyramid. The added levels
combine top-down graph context, AdaIN statistics, spatial style tokens, and graph-local
generation before decoding the transformed graph features to RGB.

M2GFT therefore strengthens the feature-generation backbone rather than adding an
image-space refinement stage. Its training objective combines multi-level
Gram/statistics matching, saliency-guided Patch-SWD, SWD, contextual matching, sliced
color OT, structure-preserving losses, and a one-sided multi-scale luminance guard
that suppresses contiguous dark-region collapse without weakening vivid style colors.

## Results

Every comparison below renders the original scene, the FSS R41 reference path, and
M2GFT from the same Gaussian cloud with identical real COLMAP camera matrices and
resolution. M2GFT produces clearer spatial color separation and a stronger transfer of
the reference palette, while suppressing the contiguous dark-region artifacts found in
the earlier model.

![Truck and style1 comparison](docs/results/truck_style1_comparison.png)

![Horse and style44 comparison](docs/results/horse_style44_comparison.png)

![Garden and style2 comparison](docs/results/garden_style2_comparison.png)

![Train and style40 comparison](docs/results/train_style40_comparison.png)

![Train and Starry Night comparison](docs/results/train_style100_comparison.png)

![Garden and Great Wave comparison](docs/results/garden_style66_comparison.png)

![Truck and style19 comparison](docs/results/truck_style19_comparison.png)

![Horse and style0 comparison](docs/results/horse_style0_comparison.png)

## Installation

The tested environment uses Python 3.10, PyTorch 2.5.1+cu121, PyG 2.7.0, and gsplat
1.5.3. A CUDA-capable NVIDIA GPU is recommended.

```bash
conda create -n m2gft python=3.10 -y
conda activate m2gft

python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install torch-scatter==2.1.2 torch-cluster==1.6.3 \
  -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
python -m pip install -r requirements.txt
```

gsplat compiles CUDA code on first use. If the active environment contains `nvcc` and
the CUDA runtime but gsplat cannot locate them, create the project-local CUDA toolkit
shim with:

```bash
python setup_cuda.py
```

## Acknowledgements

M2GFT originates from and extends
[FastSplatStyler](https://github.com/davidmhart/FastSplatStyler). We thank its authors
for publishing the graph-based Gaussian style-transfer pipeline and SelectionConv
implementation. This project also uses ideas and adapted components from
[Interpolated SelectionConv](https://github.com/davidmhart/interpolated-selectionconv)
and [LinearStyleTransfer](https://github.com/sunshineatnoon/LinearStyleTransfer).

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the license texts under
`third_party/` for details. The source code in this repository is released under the
[MIT License](LICENSE).
