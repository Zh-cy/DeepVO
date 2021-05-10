# DeepVO
Keras implementation of DeepVO:Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks

## A. Basic Information
This is the paper reproduce part of my master thesis in MRT KIT (2019-2020)  
Paper  
DeepVO: Towards End-to-End Visual Odometry with Deep Recurrent Convolutional Neural Networks  

## B. Structure
### data
euler_angle: generated euler angles (left hand, angle in degree) right hand system also works if you want  
poses: ground truth provided by KITTI  
rel_mat: generated relative pose matrix  

### KITTI_odometry_evaluation_tool
origin: https://github.com/LeoQLi/KITTI_odometry_evaluation_tool  

this is to to evaluate results quantitatively  
changes:  
KITTI_odometry_evaluation_tool/evaluation.py  
line31 change that folder  
line657 change that folder  

### dataset.py
chose your sequences to train or to test  
line33  
line134  

### dp_train.py
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


### dp_pre.py
before training make a prediction folder  1_lstm, 2_lstm... (use a different path compared to the path where you save your checkpoints, but same folder name)  
chose the best one or best several weights, move them into your prediction folder  
line18 change the folder you wanna save your predicted results  
  
### evaluation.py 
line42 - line54  
  
### loss_plot.py
first download the .csv files from tensorboard    
line41 - line45  
  
### model.py
download the pretrained weights, attention this is the file that I convert from pytorch into Keras, originally it is for  
pytorch, but here this is the same, if you are interested, you can do it by your self  
line54 change the path  
  
### plot.py
plot the results and pose transfer
line131  
line158-line163  
  
## C.Environment  
IDE pycharm  
python 3.6  
Ubuntu 18.04  
Cuda 10.2, V10.2.89  
  
## D. Packages  
package - version - latest version(2020)  
Keras	2.3.1	2.4.3  
Keras-Applications	1.0.8	1.0.8  
Keras-Preprocessing	1.1.0	1.1.2  
Markdown	3.1.1	3.3.3  
Pillow	6.2.1	8.0.1  
PyYAML	5.1.2	5.3.1  
Pygments	2.4.2	2.7.2  
Werkzeug	0.16.0	1.0.1  
absl-py	0.8.1	0.11.0  
argcomplete	1.10.0	1.12.2  
astor	0.8.0	0.8.1  
colorama	0.4.1	0.4.4  
cycler	0.10.0	0.10.0  
evo	1.5.6	1.13.0  
future	0.18.2	0.18.2  
gast	0.3.2	0.4.0  
google-pasta	0.1.7	0.2.0  
grpcio	1.24.1	1.33.2  
h5py	2.10.0	3.1.0  
joblib	0.14.0	0.17.0  
kiwisolver	1.1.0	1.3.1  
matplotlib	3.1.1	3.3.3  
natsort	6.0.0	7.1.0  
numpy	1.17.3	1.19.4  
onnx	1.7.0	1.8.0  
onnx2keras	0.0.22 0.0.24  
opencv-python	4.1.1.26	4.4.0.46  
pandas	0.25.2	1.1.4  
pip	19.0.3	20.2.4  
protobuf	3.10.0	3.14.0  
pydot	1.4.1	1.4.1  
pyparsing	2.4.2	2.4.7  
pyquaternion	0.9.5	0.9.9  
python-dateutil	2.8.0	2.8.1  
pytorch2keras	0.2.4	0.2.4  
pytz	2019.3	2020.4  
scikit-learn	0.21.3	0.23.2  
scipy	1.3.1	1.5.4  
seaborn	0.9.0	0.11.0  
setuptools	41.4.0	50.3.2  
six	1.12.0	1.15.0  
sklearn	0.0	0.0  
tensorboard	1.14.0	2.4.0  
tensorflow	1.14.0	2.3.1  
tensorflow-estimator	1.14.0	2.3.0  
tensorflow-gpu	1.14.0	2.3.1  
termcolor	1.1.0	1.1.0  
torch	1.6.0	1.7.0  
torchvision	0.7.0	0.8.1  
typing-extensions	3.7.4.2	3.7.4.3  
wheel	0.33.6	0.35.1  
wrapt	1.11.2	1.12.1  
  
Experiment a: DeepVO (baseline, reproduction work)  
Experiment b (control group, module1 test work)  
Experiment c (control group, module2 test work)  
Experiment d (CCL-VO, final work)  
