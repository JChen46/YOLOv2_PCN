# YOLOv2 with PCN in PyTorch

**DISCLAIMER:**
This is an implementation of [longcw's version](https://github.com/longcw/yolo2-pytorch)
of YOLOv2 in PyTorch. Much of the code is from that project.

This is a [PyTorch](https://github.com/pytorch/pytorch)
implementation of YOLOv2.
This project is mainly based on [darkflow](https://github.com/thtrieu/darkflow)
and [darknet](https://github.com/pjreddie/darknet).

I used a Cython extension for postprocessing and 
`multiprocessing.Pool` for image preprocessing.
Testing an image in VOC2007 costs about 13~20ms.

For details about YOLO and YOLOv2 please refer to their [project page](https://pjreddie.com/darknet/yolo/) 
and the [paper](https://arxiv.org/abs/1612.08242):
*YOLO9000: Better, Faster, Stronger by Joseph Redmon and Ali Farhadi*.

**NOTE 1:**
This is still an experimental project.
VOC07 test mAP is about 0.71 (trained on VOC07+12 trainval,
reported by [@cory8249](https://github.com/longcw/yolo2-pytorch/issues/23)).
See [issue1](https://github.com/longcw/yolo2-pytorch/issues/1) 
and [issue23](https://github.com/longcw/yolo2-pytorch/issues/23)
for more details about training.

**NOTE 2:**
I recommend to write your own dataloader using [torch.utils.data.Dataset](http://pytorch.org/docs/data.html)
since `multiprocessing.Pool.imap` won't stop even there is no enough memory space. 
An example of `dataloader` for VOCDataset: [issue71](https://github.com/longcw/yolo2-pytorch/issues/71).

**NOTE 3: (ALREADY IMPLEMENTED)**
Upgrade to PyTorch 0.4: https://github.com/longcw/yolo2-pytorch/issues/59



## Installation

1. Download the trained model [yolo-voc.weights.h5](https://drive.google.com/open?id=0B4pXCfnYmG1WUUdtRHNnLWdaMEU) 
and set the model path in `demo.py`
2. Run demo `python demo.py`. Or run `train.py` or `test.py`.

## Training YOLOv2
You can train YOLO2 on any dataset. Here we train it on VOC2007/2012.

1. Download the training, validation, test data and VOCdevkit

    ```bash
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
    ```

2. Extract all of these tars into one directory named `VOCdevkit`

    ```bash
    tar xvf VOCtrainval_06-Nov-2007.tar
    tar xvf VOCtest_06-Nov-2007.tar
    tar xvf VOCdevkit_08-Jun-2007.tar
    ```

3. It should have this basic structure

    ```bash
    $VOCdevkit/                           # development kit
    $VOCdevkit/VOCcode/                   # VOC utility code
    $VOCdevkit/VOC2007                    # image sets, annotations, etc.
    # ... and several other directories ...
    ```
    
4. Since the program loading the data in `yolo2-pytorch/data` by default,
you can set the data path as following.
    ```bash
    cd yolo2-pytorch
    mkdir data
    cd data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    
5. Download the [pretrained darknet19 model](https://drive.google.com/file/d/0B4pXCfnYmG1WRG52enNpcV80aDg/view?usp=sharing)
and set the path in `yolo2-pytorch/cfgs/exps/darknet19_exp1.py`.

6. Run the training program: `python train.py`.


For training on own data, refer to original [longcw's version](https://github.com/longcw/yolo2-pytorch).
