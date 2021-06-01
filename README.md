# DeepVO
Keras implementation of DeepVO:Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks

Basic Information
------------------------------
This is the paper reproduce part of my master thesis in MRT KIT (2019-2020)  

**Original Paper**:

[DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks](https://arxiv.org/abs/1709.08429)

About my Work
------------------------------
* ### Structure
<div align=left><img src="https://raw.githubusercontent.com/Zh-cy/images/master/DeepVO/cclvo.png" height="415" width="855"/> </div> 

* ### Demo (compared with VISO-Mono)


Details
------------------------------
* ### data
euler_angle: generated euler angles (left hand, angle in degree) right hand system also works if you want  
poses: ground truth provided by KITTI  
rel_mat: generated relative pose matrix  

* ### KITTI_odometry_evaluation_tool
origin: https://github.com/LeoQLi/KITTI_odometry_evaluation_tool  

this is to to evaluate results quantitatively  
changes:  
KITTI_odometry_evaluation_tool/evaluation.py  
line31 change that folder  
line657 change that folder  

* ### dataset.py
chose your sequences to train or to test  
line33  
line134  

* ### dp_train.py
before training make a checkpoint saving folder to save the checkpoints like 1_lstm, 2_lstm...  
I have also tried bidirectional lstm, but it doesn't converge. I didn't delete this module, so you can still make a test  
  
read the parser arguments before training  
examples of different mode:  
a.train from start  
python3 dp_train.py --folder 1 --lr_base 0.0001 --epoch_max 100 --epoch_lr_decay 100  
                    --initial_epoch 1 --dropout 0.5 --beta 50 --gpu 2  
(this is the option that I used)  

b.resume training (eg. interrupted at epoch 45)  
python3 dp_train.py --folder 1 --lr_base 0.0001 --epoch_max 100 --epoch_lr_decay 100 --resume 1  
                    --initial_epoch 45 --cnn_lstm_weights cnn_lstm_weight.45-x.xxxx.h5 --dropout 0.5 --beta 50 --gpu 2  
line39 change tensorboard folder  
line54 change input images folder  
linie56 change ground truth euler angles folder  
line70 change the folder you wanna save your weights -- checkpoint_path  


* ### dp_pre.py
before training make a prediction folder  1_lstm, 2_lstm... (use a different path compared to the path where you save your checkpoints, but same folder name)  
chose the best one or best several weights, move them into your prediction folder  
line18 change the folder you wanna save your predicted results  
  
* ### evaluation.py 
line42 - line54  
  
* ### loss_plot.py
first download the .csv files from tensorboard    
line41 - line45  
  
* ### model.py
download the pretrained weights, attention this is the file that I convert from pytorch into Keras, originally it is for  
pytorch, but here this is the same, if you are interested, you can do it by your self  
line54 change the path  
  
* ### plot.py
plot the results and pose transfer
line131  
line158-line163  
  
Environment  
------------------------------
IDE pycharm  
python 3.6  
Ubuntu 18.04  
Cuda 10.2, V10.2.89  

Keras	2.3.1  
Pillow	6.2.1  
matplotlib	3.1.1  
numpy	1.17.3  
opencv-python	4.1.1.26  
pip	19.0.3  
pytorch2keras	0.2.4  
tensorboard	1.14.0  
tensorflow	1.14.0  

Experiments  
------------------------------
Experiment a: DeepVO (baseline, reproduction work)  
Experiment b (control group, module1 test work)  
Experiment c (control group, module2 test work)  
Experiment d (CCL-VO, final work)  
