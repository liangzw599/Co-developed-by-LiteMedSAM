# LiteMedSAM

A lightweight version of MedSAM for fast training and inference. The model was trained with the following two states:

- Stage 1. Distill a lightweight image encoder `TinyViT` from the MedSAM image encoder `ViT` by imposing the image embedding outputs to be the same
- State 2. Replace the MedSAM image encoder `ViT` with `TinyViT` and fine-tune the whole pipeline

# Obtained training test results

- [The best model](https://pan.baidu.com/s/11Cs1hOmGBaPWtf3BBvFo8w?pwd=1111) can be downloaded.

# Sanity test

- Run the following command for a sanity test.

```bash
python CVPR24_LiteMedSAM_infer_v2.py -i test_demo/imgs/ -o test_demo/segs
```
We have improved the original code to save the process of model transformation and shorten the test time.


# Installation

The official recommendation is tested with: `Ubuntu 20.04` | Python `3.10` | `CUDA 11.8` | `Pytorch 2.1.2`
Also, we have tested with: `CentOS 7.9` | Python `3.10.13` | `CUDA 12.2` | `Pytorch 2.1.2`

1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone -b LiteMedSAM https://github.com/bowang-lab/MedSAM/`
4. Enter the MedSAM folder `cd MedSAM` and run `pip install -e .`


# Model Training

## Data preprocessing
1. Download the Lite-MedSAM [checkpoint](https://drive.google.com/file/d/18Zed-TUTsmr2zc5CHUWd5Tu13nb6vq6z/view?usp=sharing) and put it under the current directory.
2. The data set is divided into the training set and the test set according to the ratio of 4:1.
3. The ability to convert training data from 'npz' to 'npy' format was added to the training file: `train_one_gpu.py`

## Loss function
1. `Boundary loss`is newly introduced.
2. `AutomaticWeightedLoss` is added to adjust the weight of each loss function by means of adaptation.

The definitions of `Boundary loss` and `AutomaticWeightedLoss` can be viewed at `loss_op.py`


## Single GPU

We trained Lite-MedSAM on a single GPU, run:
```bash
python train_one_gpu.py \
    -data_root data/MedSAM_train \
    -pretrained_checkpoint lite_medsam.pth \
    -work_dir work_dir \
    -num_workers 4 \
    -batch_size 4 \
    -num_epochs 10
```

To resume interrupted training from a checkpoint, run:
```bash
python train_one_gpu.py \
    -data_root data/MedSAM_train \
    -resume work_dir/medsam_lite_latest.pth \
    -work_dir work_dir \
    -num_workers 4 \
    -batch_size 4 \
    -num_epochs 10
```

For additional command line arguments, see `python train_one_gpu.py -h`.

# Acknowledgements
We thank the authors of [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) and [TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT) for making their source code publicly available.







