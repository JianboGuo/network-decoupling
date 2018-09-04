# [Network Decoupling: From Regular to Depthwise Separable Convolutions](https://arxiv.org/abs/1808.05517)

**BMVC 2018**, by [Jianbo Guo](https://jianboguo.github.io/), Yuxi Li, [Weiyao Lin](https://weiyaolin.github.io/), [Yurong Chen](https://scholar.google.com/citations?user=MKRyHXsAAAAJ&hl=en) and [Jianguo Li](https://sites.google.com/site/leeplus/).

In this repository, we released code for the experiments in the above paper which are all about different deploy time network optimization algorithm.
- network decoupling (ND)
- spatial decomposition (SD)
- channel decomposition (CD)

The current network supported: VGG-16, ResNet series, DenseNet series and AlexNet-bn.
    
### Contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#channel-pruning) 
4. [Experiment Result](#experiment-results) 
4. [Reference](#reference)

### Requirements
1. Python3 packages you might not have: `scipy`, `sklearn`, `easydict`, use `sudo pip3 install` to install.
2. An NVIDIA GPU is recommanded.

### Installation
1. Clone the repository of Caffe and compile it
```Shell
    git clone https://github.com/BVLC/caffe.git
    cd caffe
    # modify Makefile.config to the path of the library on your machine, please make sure the python3 interface is supported
    make -j8
    make pycaffe
```
2. Clone this repository 
```Shell
    https://github.com/JianboGuo/network-decoupling.git
```
    
### Usage  
1. Download the original model files (.prototxt and .caffemodel) and move them to the directory of `models`

2. Make proper configurations in `config.py`
   To make sure the network optimization works well, please enter the file `config.py` and change the configuration of the parameters according to the comment above them.

   Note, among the hyperparameters above the `SD_Param`,`ND_Param`,`CD_Param` and `device_id` could also be specified in command line (see section 3), while other parameters must be set correctly according to the comment above.

3. Command Line Usage
To decouple a network, use the following command
```Shell
    python3.5 main.py <optional arguments>
    optional arguments:
        -h, --help            show this help message and exit
        -sd                   enable spatial decomposition
        -nd                   enable network decoupling
        -cd                   enable channel decompostion
        -data                 enable data driven for spatial decomposition
        -speed SPEED          sd speed up ratio of spatial decomposition
        -threshold THRESHOLD  energy threshold for network decouple
        -gpu GPU              caffe devices
        -model MODEL          caffe prototxt file path
        -weight WEIGHT        caffemodel file path
        -action {decomp,compute,test}
                              compute, test or decompose the model
        -iterations ITER      test iterations
        -rank RANK            rank for network decoupling
        -DP                   flags to set DW + PW decouple (default is PW + DW)

```

For example, suppose the VGG-16 network is in folder `models/` and named as `vgg.prototxt` and `vgg.caffemodel`, you could use the following command to conduct network decoupling (ND) with threshold 0.95:
```Shell
    python3.5 main.py -model models/vgg.prototxt -weight models/vgg.caffemodel -action -decomp -nd -threshold 0.95
```
Or you can decouple the network using a given rank instead of threshold:
```Shell
    python3.5 main.py -model models/vgg.prototxt -weight models/vgg.caffemodel -action -decomp -nd -rank 5
```
Similarly, you could decompose the model with both network decoupling and spatial decomposition with compression ratio of 2 and data reconstruction:
```Shell
    python3.5 main.py -model models/vgg.prototxt -weight models/vgg.caffemodel -action -decomp -nd -rank 5 -sd -speed 2.0 -data
```

Note: the decomposition results are saved under the directory of `models`, with the name format of `new_xx_relu_separ_xx.prototxt`, where the first `xx` is the combination string of compression ratio and/or ND threshold, and the second `xx` is the original model name. The result weights file is also saved under `models` with the name format of `new_decomp_merged_xx.caffemodel` where `xx` is the original model name of input. 

When the `action` FLAG is `compute`, the program will compute the FLOPs of specified model. When the `action` FLAG is `test`, the program will test the model accuracy on specified dataset.

`model`,`weight` and `action` must be specified in command line. As for other arguments, if they are not specified in command line, the system will use the default value in configuration file `config.py` (see section 2).

The combinations of sd + cd and sd + nd + cd are not supported now.

### Experiment Result
Acceleration performance on VGG16 with (a) single method and (b) combined method. “with ND" means the experiments are conducted in combination with our network decoupling (ND). The models are tuned with fixed 1.0% top-5 accuracy drop. (a) also lists the corresponding top-1 accuracy drops.

(a) Single methods:

| Method | Decomposed FLOPs | top-1 drop(%) |
| ------ | ------|------|
|ND| 8.61G | 1.55 |
|CP| 9.89G | 1.68 |
|SD| 7.20G | 1.96 |
|CD| 6.52G | 2.1 |


(b) Combined methods:

| Method | FLOPs without ND | FLOPs with ND (ND+X) |
| ------ | ------|------|
|CD| 6.52G | 4.72G |
|SD| 7.20G | **4.15G** |
|CP| 9.89G | 8.49G |
|CD+SD| 4.32G | 4.28G |
|CD+CP| 7.16G | 5.45G |
|CD+SD+CP | 4.92G | 4.70G|

More results could be found in [our paper](https://arxiv.org/abs/1808.05517)



### Related Repository
This project is built on the repository of [channel pruning](https://github.com/yihui-he/channel-pruning), many thanks to the contributor of this work.

### Reference

This work is based on our work *Network Decoupling: From Regular to Depthwise Separable Convolutions (BMVC2018)*. If you think this is helpful for your research, please consider append following bibtex config in your latex file.

```Latex
@inproceedings{guo2018nd,
  title = {Network Decoupling: From Regular to Depthwise Separable Convolutions},
  author = {Guo, Jianbo and Li, Yuxi and Lin, Weiyao and Chen, Yurong and Li, Jianguo},
  booktitle = {BMVC},
  year = {2018}
}
```

This repository is also referenced to the work of channel decomposition and spatial decomposition, to know more details about channel decomposition and spatial decomposition, please refer to following papers.
- [Speeding up Convolutional Neural Networks with Low Rank Expansions](https://arxiv.org/abs/1405.3866)
- [Accelerating Very Deep Convolutional Networks for Classification and Detection](https://arxiv.org/abs/1505.06798)
