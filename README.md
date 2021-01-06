# yolo_quantization
![GitHub stars](https://img.shields.io/github/stars/ArtyZe/yolo_quantization) ![GitHub forks](https://img.shields.io/github/forks/ArtyZe/yolo_quantization)  ![GitHub watchers](https://img.shields.io/github/watchers/ArtyZe/yolo_quantization)

![](https://img.shields.io/badge/Linux CPU-Pass-blue.svg)

The code is to quantization **float32 network** of darknet to **uint8 network** based of paper:

>**Quantization and Training of Neural Networks for Efficient**

< https://arxiv.org/abs/1712.05877 >

[The Commond to Run My Project]
=========
[Linux]
Train: 
>**set GPU=1 in Makefile**

	make -j8

	./darknet detector train cfg/voc_nok.data cfg/yolov3-tiny-mask_quant.cfg [pretrain weights file I gave to you(default in cfg folder)]

Test:
>**set GPU=0 in Makefile**
	
	make -j8
	
	./darknet detector test cfg/voc_nok.data cfg/yolov3-tiny_quant.cfg [weights file] [image path]

[Windows]
Test:

	yolo_quantization detector test [abs path to data file] [abs path to cfg file] [abs path to weights file] [abs path to image file]

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


