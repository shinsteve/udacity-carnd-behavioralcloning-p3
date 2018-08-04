# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/history_before.png "Learning history it is over-fitting"
[image2]: ./images/history_with_dropout.png "Learning history with Drop out layer"
[image3]: ./images/left.jpg "Left camera image"
[image4]: ./images/center.jpg "Center camera image"
[image5]: ./images/right.jpg "Right camera image"
[image6]: ./images/center.jpg "Normal Image"
[image7]: ./images/flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network followed by fully connected layer, which is based on NVIDIA's work which is illustrated in the lesson material.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (code: function build_network()).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code: function load_dataset()). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (code: function build_network()).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, etc.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model which is NVIDIA's work as illustrated in the lesson material.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 
After introducing Drop-out layer, that over-fitting was resolved as shown in the below graph.

![Learning history: overfitting][image1]

The final step was to run the simulator to see how well the car was driving around track one.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

![Learning history: with Dropout][image2]

#### 2. Final Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 RGB image   					| 
| Cropping         	| 50 lines in top and 20 lines bottom are removed	| 
| Normalize and 0 mean  | lambda x: x / 127.5 - 1.0   					| 
| Convolution 5x5     	| 24 features  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  				|
| Dropout				| Drop rate = 20 %								|
| Convolution 5x5     	| 36 features  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  				|
| Dropout				| Drop rate = 20 %								|
| Convolution 5x5     	| 48 features  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  				|
| Dropout				| Drop rate = 20 %								|
| Convolution 5x5     	| 64 features  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  				|
| Dropout				| Drop rate = 20 %								|
| Convolution 5x5     	| 64 features  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  				|
| Dropout				| Drop rate = 20 %								|
| Flattern				| 									|
| Fully connected		| outputs 100 									|
| Dropout				| Drop rate = 40 %								|
| Fully connected		| outputs 50 									|
| Dropout				| Drop rate = 40 %								|
| Fully connected		| outputs 16 									|
| Fully connected		| outputs 1 									|
 

#### 3. Creation of the Training Set & Training Process

I recorded the below 3 types of driving for a several laps each.

1. Center lane driving as nomarl case
1. Zig-Zag driving as recovery driving from one side back to center
1. The above driving in opposite direction of the track for the generalization in learning

I used the below 2 simulators for different purpose described below.

1. Normal simulator to capture not only center camera image but also left and right camera images that can be ussed as augmented data of center image. Note that the beta simulator captures only center camera image.
1. Beta simulator to get the angles which is recorded in the driving by mouse-dragging, which enables more fine control than by keyboard.

The following images show left, center, and right camera images

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would be opposite sign with the same value. For example, the right side of below image is an image that has then been flipped:

![original][image6]
![flipped][image7]

Regarding the left and right camera image, the corresponding angles is corrected to add/subract some value (=0.2) so that it gets as if it is a center image.

Finally I got the following number of data set.
  * center: 22491, left/right: 8769 each

| Images captured in | Number |
|:---------:|:-----:|
| Center cam | 8769 |
| Center cam flipped | 8769 |
| Center cam (beta sim) | 13722 |
| Center cam flipped (beta sim) | 13722 |
| Left cam | 8769 |
| Left cam flipped | 8769 |
| Right cam | 8769 |
| Right cam flipped | 8769 |
| **Total** | **80058** |

Training process was performed with the following condition:

* 20 % of the captured data is separated as validation set to determine if the model was over or under fitting.
* Epochs : 10 times
* Adam optimizer is used so that manually training the learning rate wasn't necessary.
