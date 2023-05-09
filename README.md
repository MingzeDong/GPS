# GPS (Graph rewiring via propensity score)

This repository includes the GPS (graph propensity score) part for paper "Towards Understanding and Reducing Graph Structural Noise for GNNs".

<img width="1442" alt="image" src="https://github.com/MingzeDong/GPS/assets/68533876/e510e851-2698-4b80-bf83-2fbe7bbedc24">


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

The UMAP visualization of different rewired graph embeddings for the Cornell dataset:

<img width="1854" alt="image" src="https://user-images.githubusercontent.com/68533876/227424482-0f5cf405-5e1a-4486-84e5-92e4eed8155a.png">
