# yolo_quantization
![GitHub stars](https://img.shields.io/github/stars/ArtyZe/yolo_quantization) ![GitHub forks](https://img.shields.io/github/forks/ArtyZe/yolo_quantization)  ![GitHub watchers](https://img.shields.io/github/watchers/ArtyZe/yolo_quantization)

![](https://img.shields.io/badge/LinuxCPU-Pass-brightgreen.svg?style=plastic) ![](https://img.shields.io/badge/LinuxGPU-Pass-brightgreen.svg?style=plastic) ![](https://img.shields.io/badge/WindowsCPU-Pass-brightgreen.svg?style=plastic)

The code is to quantization **float32 network** of darknet to **uint8 network** based of paper:

>**Quantization and Training of Neural Networks for Efficient**

< https://arxiv.org/abs/1712.05877 >

[Requirments]
=========
MKL:

If you want to use mkl to accelerate, you need to install mkl by yourself, else do not need to install mkl

1. Download and Install MKL:	
	https://pan.baidu.com/s/1vl8W7gp1MS_E_owgc6zrkA
	password: v37i
2. Fit MKL
	1. For Linux, change `MKLROOT` in Makefile to your own mkl install path
	2. For Windows, fit mkl path in setting of vs, follow this blog if you have no experience: https://www.cnblogs.com/Mayfly-nymph/p/11617651.html

[The Commond to Run My Project]
=========
[Linux]
Train: 
>**set GPU=1 in Makefile**

	make -j8

	./darknet detector train cfg/voc_nok.data cfg/yolov3-tiny-mask_quant.cfg [pretrain weights file I gave to you(default in cfg folder)]

[Linux] Test:
>**set GPU=0, QUANTIZATION=1 in Makefile and OPENBLAS=1 if you use mkl**
	make -j8
	
	./darknet detector test cfg/voc_nok.data cfg/yolov3-tiny_quant.cfg [weights file] [image path]

[Windows]   
Test:
	
    1. close macro OPENBLAS in vs, else open OPENBLAS to use mkl

    2. yolo_quantization.exe detector test [abs path to data file] [abs path to cfg file] [abs path to weights file] [abs path to image file]

[Pretrain Cfg file and Weights file]
=========
	https://pan.baidu.com/s/16_ULXdNPmIhoEmu7jXmkmQ 
	password: qy8a 
	
[Performance]
=========
 | quantization | inference time (intel chip 64bit) | recall | precision | f1 score |
 | :------: | :------: | :------: | :------: | :------: |
 | darknet | 0.83s | 74.43 | 89.45 | 81.25| 
 | quantization mine | 0.34s | 90.08 | 91.83 | 90.94 |


