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

[Pretrain Cfg file and Weights file]
=========
   https://pan.baidu.com/s/1Pq5Drj8t9UEKTYNJszjJSQ 
   password: waga

[Performance]
=========
 | quantization | inference time (intel chip 64bit) | recall | precision | f1 score |
 | :------: | :------: | :------: | :------: | :------: |
 | darknet | 0.92s | 66.79 | 84.13 | 74.47| 
 | quantization mine | 0.55s | 57.25 | 77.72 | 66.52 |


