# EFINet: Restoration for Low-light Images via Enhancement-Fusion Iterative Network
## Pytorch
Pytorch implementation of EFINet.
<img src=data/EFINet_architecture.png width="50%"/>

## Requirements

```
python 3.7  
torch 1.4.0  
torchvision 0.2.1  
cuda 10.1  
numpy  
opencv
```

## Usage

The test results will be saved at `data/result`.

```key
python lowlight_test.py
```

The script processes images located in the subfolders of the `test_data` directory and creates a new `result` folder within the `data` directory. The enhanced images will be available in the `result` folder.

## Citation

If you find this work useful for your research, please consider citing our paper

```
@ARTICLE{liu2022efinet,
  title={EFINet: Restoration for Low-light Images via Enhancement-Fusion Iterative Network}, 
  author={Liu, Chunxiao and Wu, Fanding and Wang, Xun},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  year={2022},
  volume={32},
  number={12},
  pages={8486-8499},
  doi={10.1109/TCSVT.2022.3195996}}
```
