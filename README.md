# GPS (Graph rewiring via propensity score)

This repository includes the GPS (graph propensity score) part for paper "Towards Understanding and Reducing Graph Structural Noise for GNNs".

<img width="1235" alt="image" src="https://github.com/MingzeDong/GPS/assets/68533876/1b7e24d9-cdc3-4773-a375-d847a7d5aaa8">



## Dependencies

    numpy
    torch==1.13.0
    matplotlib
    sklearn
    scipy
    numba (for SDRF)
    torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
    torch-geometric==2.2.0
    ogb==1.3.5
    networkx==2.6.3
    Ray Tune

## Results
The UMAP visualization of different rewired graph embeddings for the Cornell dataset:

<img width="1854" alt="image" src="https://user-images.githubusercontent.com/68533876/227424482-0f5cf405-5e1a-4486-84e5-92e4eed8155a.png">

For more details, please refer to our paper: [Towards Understanding and Reducing Graph Structural Noise for GNNs](https://proceedings.mlr.press/v202/dong23a.html)
```
@InProceedings{pmlr-v202-dong23a,
  title = 	 {Towards Understanding and Reducing Graph Structural Noise for {GNN}s},
  author =       {Dong, Mingze and Kluger, Yuval},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {8202--8226},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR}
}
```
