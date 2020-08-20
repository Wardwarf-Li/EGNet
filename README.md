# EGNet
This implementation mainly refers to (https://github.com/JXingZhao/EGNet)

Use the sal2edge.m to generate the edge label for training.
### For training:
1. Download [training data](https://pan.baidu.com/s/1LaQoNRS8-11V7grAfFiHCg) (fsex) ([google drive](https://drive.google.com/open?id=1wduPbFMkxB_3W72LvJckD7N0hWbXsKsj));

2. Download [initial model](https://pan.baidu.com/s/1dD2JOY_FBSLzjp5tUPBDBQ) (8ir7) ([google_drive](https://drive.google.com/open?id=1q7FtHWoarRzGNQQXTn9t7QSR8jJL8vk6)); 

3. Change the image path and intial model path in run.py and dataset.py;

4. Start to train with `python3 run.py --mode train`.

### For testing:
1. Download [pretrained model](https://pan.baidu.com/s/1s35ZyGDSNVzVIeVd7Aot0Q) (2cf5)  ([google drive](https://drive.google.com/open?id=17Ffc6V5EiujtcFKupsJXhtlQ3cLK5OGp)); 

2. Generate saliency maps and evaluation metric for SOD dataset by `python3 run.py --mode test`






