# Third-party notices

M2GFT builds on the graph representation and SelectionConv workflow introduced
by [FastSplatStyler](https://github.com/davidmhart/FastSplatStyler) and
[Interpolated SelectionConv](https://github.com/davidmhart/interpolated-selectionconv).
The adapted SelectionConv and pooling implementation is covered by the
[FastSplatStyler MIT License](third_party/fast_splat_styler/LICENSE).

The R41 encoder, decoder, transformation design, and local pretrained assets are
based on [LinearStyleTransfer](https://github.com/sunshineatnoon/LinearStyleTransfer)
and retain its [BSD 2-Clause License](third_party/linear_style_transfer/LICENSE).

M2GFT's multi-level r31/r21/r11 graph feature generator, spatial style-token
conditioning, training losses, Gaussian interpolation, and experiment pipeline are
provided as part of this project.
