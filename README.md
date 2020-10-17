# yolo_quantization
The code is to quantization **float32 network** of darknet to **uint8 network** based of paper:

>**Quantization and Training of Neural Networks for Efficient**

< https://arxiv.org/abs/1712.05877 >

[The Commond to Run My Project]
=========
Train: 

>**set GPU=1 in Makefile**

	make -j8

	./darknet segmenter train cfg/voc_nok.data cfg/yolov3-tiny-mask_quant.cfg [pretrain weights file I gave to you]

Test:
>**set GPU=0 in Makefile**
	
	make -j8
	
	./darknet segmenter test cfg/voc_nok.data cfg/yolov3-tiny-mask_quant.cfg [weights file] [image path]

[Pretrain weights file and cfg file]  
========  

1. https://www.dropbox.com/sh/9mv29eoy9fa35ie/AAAq-53zo8NuD3cjzWFZfMboa?dl=0
2. https://pan.baidu.com/s/15gcrXGzb-fY2vGdl4KlLqg
   password: bk01

