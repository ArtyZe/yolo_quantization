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
	
	./darknet detector test cfg/voc_nok.data cfg/yolov3-tiny_quant.cfg [weights file] [image path]

[Pretrain weights file and cfg file]  
========  

[Pretrain Cfg file and Weights file]
=========
   https://pan.baidu.com/s/1Pq5Drj8t9UEKTYNJszjJSQ 
   password: waga

[Runtime per image]
=========
 | quantization | inference time (intel chip 64bit) | recall score | f1 score |
 | :------: | :------: | :------: | :------: |
 | darknet | 0.92s | 67.18 | 55.00 |
 | quantization with kl | 0.72s | 65.51 | 53.26 |
 | quantization mine | 0.55s | 56.87 | 50.68 |


