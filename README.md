# EFINet: Restoration for Low-light Images via Enhancement-Fusion Iterative Network
## Pytorch
Pytorch implementation of EFINet.

## Requirements
python 3.7
torch 1.4.0
torchvision 0.2.1
cuda 10.1
numpy
opencv

## Usage
The test results will be saved at "data/result".
```key
python lowlight_test.py
```  
The script will process the images in the sub-folders of "test_data" folder and make a new folder "result" in the "data". You can find the enhanced images in the "result" folder.
