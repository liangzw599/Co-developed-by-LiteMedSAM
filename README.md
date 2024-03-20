# LiteMedSAM

A lightweight version of MedSAM for fast training and inference. The model was trained with the following two states:

- Stage 1. Distill a lightweight image encoder `TinyViT` from the MedSAM image encoder `ViT` by imposing the image embedding outputs to be the same
- State 2. Replace the MedSAM image encoder `ViT` with `TinyViT` and fine-tune the whole pipeline

# Obtained training test results

- The model has been trained for 41 epochs in the early stage, and [the trained model](https://pan.baidu.com/s/118DLCjvOycXFNuaEKrqr_g?pwd=1111) can be downloaded and place in  it at e.g., `MedSAM/workdir`



## Installation

The official recommendation is tested with: `Ubuntu 20.04` | Python `3.10` | `CUDA 11.8` | `Pytorch 2.1.2`
Also, we have tested with: `CentOS 7.9` | Python `3.10.13` | `CUDA 12.2` | `Pytorch 2.1.2`

1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone -b LiteMedSAM https://github.com/bowang-lab/MedSAM/`
4. Enter the MedSAM folder `cd MedSAM` and run `pip install -e .`






