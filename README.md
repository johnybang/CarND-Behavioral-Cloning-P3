# **Behavioral Cloning** 
John Bang, 4/18/17

## Introduction

### As part of the Udacity Self Driving Car Engineer Nanodegree program we train a deep neural network to predict the proper steering angle for driving a simulated autonomous vehicle around a simulated driving track.  This project is a practical exercise in deep learning for a regression task and thoughtful data collection.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/good_driving_center.jpg "Good Driving Center Image"
[image2]: ./examples/good_driving_left.jpg "Good Driving Left Image"
[image3]: ./examples/good_driving_right.jpg "Good Driving Right Image"
[image4]: ./examples/recovery1.jpg "Recovery Image 1"
[image5]: ./examples/recovery2.jpg "Recovery Image 2"
[image6]: ./examples/recovery3.jpg "Recovery Image 3"

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
* video.mp4 containing a video of the car driving autonomously around the track for one lap

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model architecture is approximately equivalent to [this Nvidia architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars):

![Nvidia Architecture](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

My model has an input image size of 160x320 (in contrast to Nvidia's 66x200 image size). I normalize the images using a Keras lambda layer at code line 62.  It then crops away horizon and car hood portions at code line 63.

Although I did a couple tests with LeNet5 architecture at code line 70, I moved on to the Nvidia architecture pretty early.  My motivation was to use the architecture that had explicitly been used for autonomous driving, keeping in mind that I likely need to deal with its complexity by making sure to regularize.

The Nvidia architecture code starts at line 85. The Nvidia inspired convolutional layers use a stride of 2x2 with valid padding for the 5x5 filters as implied by the layer dimensions shown in the diagram ((previous_dimension - 4) / 2). The 3x3 filters are a stride of 1 with valid padding.  I use rectified linear unit activation for all of the convolutional layers to introduce nonlinearities (important for the model's expressive potential) and linear activation for the fully connected layers since it is a regression task. (In the future, I may consider experimenting with adding relu nonlinearity to some or all of the fully connected layers.)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers and L2 regularization in order to reduce overfitting as seen in the Nvidia architecture code starting at line 85.  Some of these are inactive for the model parameters which trained my final model.  In my final model, I applied 50% dropout to the fully connected layers only; no regularization elsewhere.  The model training parameters are at line 53.

The model was trained and validated on different data sets with an 80%/20% split to ensure that the model was not overfitting (keyword parameter at code line 123).  In addition, the callbacks at line 120 implement Early Stopping to further prevent overfitting and automate the num_epochs tuning process.

Ultimately, the model had to be tested by running it through the simulator and visually inspecting that the vehicle could stay on the track.  (Given that this is an open-source simulator, I'm curious how of an undertaking it would be for me to try to automate the drive-quality assessment process... perhaps in a future with more time on my hands...)

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (code line 116). Dropout locations, a few L2 regularization attempts, cropping, left/right correction factor, etc. were tuned with varying experimental detail through iterative testing. Here's an example of some test logs:

| Driving Data                                    | loss_func   |   Clockwise Centered |   Clockwise Recovery |   beta_conv |   beta_fc |   keep\_prob_fc |   keep\_prob_conv | use\_lr_images   |   lr\_correction_factor | arch       | norm   | flip_center   |   flip\_left_right | crop     |   train loss |   val loss | driving behavior                                                                                                | other notes                          |
|:------------------------------------------------|:------------|---------------------:|---------------------:|------------:|----------:|---------------:|-----------------:|:----------------|-----------------------:|:-----------|:-------|:--------------|------------------:|:---------|-------------:|-----------:|:----------------------------------------------------------------------------------------------------------------|:-------------------------------------|
| Center Driving                                  | mse         |                  nan |                  nan |       0     |         0 |            1   |              1   | False           |                  nan   | 1 layer fc | False  | False         |               nan | (0,0)    |  2511        | 1321.43    | driving in circles                                                                                              | nan                                  |
| Center Driving                                  | mse         |                  nan |                  nan |       0     |         0 |            1   |              1   | False           |                  nan   | 1 layer fc | True   | False         |               nan | (0,0)    |     1.5535   |    1.13835 | driving in circles                                                                                              | nan                                  |
| Center Driving                                  | mse         |                  nan |                  nan |       0     |         0 |            1   |              1   | False           |                  nan   | LeNet5     | True   | False         |               nan | (0,0)    |     0.0093   |    0.01134 | driving fairly straight but wanders without recovery                                                            | nan                                  |
| Center Driving                                  | mse         |                  nan |                  nan |       0     |         0 |            1   |              1   | False           |                  nan   | LeNet5     | True   | True          |               nan | (0,0)    |     0.0105   |    0.01031 | driving better but still wanders without recovery                                                               | nan                                  |
| Center Driving                                  | mse         |                  nan |                  nan |       0     |         0 |            1   |              1   | True            |                    0.2 | LeNet5     | True   | True          |                 1 | FALSE    |     0.0457   |    0.02988 | veers off left into the water                                                                                   | flatten input shape bug              |
| Center Driving                                  | mse         |                  nan |                  nan |       0     |         0 |            1   |              1   | True            |                    0.2 | LeNet5     | True   | True          |                 0 | (70, 25) |     0.0129   |    0.02502 | ok until first tight curve into water                                                                           | flatten input shape bug              |
| Center Driving                                  | mse         |                  nan |                  nan |       0     |         0 |            1   |              1   | True            |                    0.2 | LeNet5     | True   | True          |                 0 | (70, 25) |     0.0113   |    0.0217  | ok until first tight curve into water                                                                           | fixed flatten input shape bug        |
| Center Driving                                  | mse         |                  nan |                  nan |       0     |         0 |            1   |              1   | True            |                    0.2 | Nvidia     | True   | True          |                 0 | (70, 25) |     0.0085   |    0.02141 | further than lenet but still into the same water                                                                | nan                                  |
| Center Driving                                  | mse         |                  nan |                  nan |       0     |         0 |            1   |              1   | False           |                  nan   | Nvidia     | True   | True          |               nan | (70, 25) |     0.0094   |    0.00807 | great until second half of s-curve into water                                                                   | nan                                  |
| John Center Driving                             | mse         |                  nan |                  nan |       0     |         0 |            1   |              1   | False           |                  nan   | Nvidia     | True   | True          |               nan | (70, 25) |     3.44e-05 |    0.00135 | off track at first curve                                                                                        | nan                                  |
| John Center Driving                             | mse         |                  nan |                  nan |       0     |         0 |            1   |              1   | True            |                  nan   | Nvidia     | True   | True          |                 0 | (70, 25) |     0.0042   |    0.02469 | rode right curb on first curve but no water                                                                     | nan                                  |
| John Center Driving                             | mse         |                  nan |                  nan |       0     |         0 |            0.5 |              1   | False           |                  nan   | Nvidia     | True   | True          |               nan | (70, 25) |     0.0075   |    0.00846 | rode right curb on straight, to forest on first slight curve                                                    | regularization causing understeering |
| John Center Driving                             | mse         |                  nan |                  nan |       0     |         0 |            0.5 |              1   | True            |                  nan   | Nvidia     | True   | True          |                 0 | (70, 25) |     0.0144   |    0.05787 | rode right curb on straight, to forest on first slight curve                                                    | regularization causing understeering |
| JohnCenter, JohnRec                             | mse         |                  nan |                  nan |       0     |         0 |            1   |              1   | False           |                  nan   | Nvidia     | True   | True          |               nan | (70, 25) |     0.0035   |    0.02749 | turned toward right edge, off road at first curve                                                               | nan                                  |
| UdacityCenter, JohnRec                          | mse         |                  nan |                  nan |       0     |         0 |            1   |              1   | False           |                  nan   | Nvidia     | True   | True          |               nan | (70, 25) |   nan        |  nan       | nan                                                                                                             | nan                                  |
| JohnNewAll                                      | mse         |                    1 |                    1 |       0     |         0 |            1   |              1   | False           |                  nan   | Nvidia     | True   | False         |               nan | (70, 25) |     0.0016   |    0.00812 | ok until big turn before bridge then stuck on right; hugging right throughout                                   | nan                                  |
| JohnNewAll                                      | mse         |                    1 |                    1 |       0     |         0 |            0.5 |              1   | False           |                  nan   | Nvidia     | True   | False         |               nan | (70, 25) |     0.0034   |    0.00898 | ok until big turn before bridge then stuck on right; hugging right throughout                                   | dropout fc, not last                 |
| JohnNewAll                                      | mse         |                    1 |                    1 |       0     |         0 |            0.5 |              1   | False           |                  nan   | Nvidia     | True   | False         |               nan | (70, 25) |     0.0055   |    0.00819 | pretty decent but got tricked by off ramp after bridge                                                          | dropout all fc                       |
| JohnNewAll                                      | mse         |                    1 |                    1 |       0     |         0 |            0.5 |              1   | False           |                  nan   | Nvidia     | True   | False         |               nan | (70, 25) |     0.0072   |    0.00891 | ok until big turn before bridge into water                                                                      | dropout all fc, refactored           |
| JohnNewAll, JohnFailCorrPostBridge              | mse         |                    1 |                    1 |       0     |         0 |            0.5 |              1   | False           |                  nan   | Nvidia     | True   | False         |               nan | (70, 25) |     0.0044   |    0.00547 | ok until big turn before bridge into water                                                                      | nan                                  |
| JohnNewAll, JohnFailCorrPostBridgePreBridge     | mse         |                    1 |                    1 |       0     |         0 |            0.5 |              1   | False           |                  nan   | Nvidia     | True   | False         |               nan | (70, 25) |     0.0046   |    0.00493 | pretty decent but got tricked by off ramp after bridge                                                          | dropout all fc                       |
| JohnNewAll, JohnFailCorrPostBridgePreBridgePlus | mse         |                    1 |                    1 |       0     |         0 |            0.5 |              1   | False           |                  nan   | Nvidia     | True   | False         |               nan | (70, 25) |     0.0038   |    0.00652 | pretty decent but got tricked by off ramp after bridge, hugged right edge at beginning                          | nan                                  |
| JohnNewAll, JohnFailCorrPostBridgePreBridgePlus | mse         |                    0 |                    0 |       0     |         0 |            0.5 |              1   | False           |                  nan   | Nvidia     | True   | False         |               nan | (70, 25) |     0.0053   |    0.00531 | ok until big turn before bridge into water                                                                      | nan                                  |
| JohnNewAll, JohnFailCorrPostBridgePreBridgePlus | mse         |                    0 |                    0 |       0     |         0 |            0.8 |              1   | False           |                  nan   | Nvidia     | True   | False         |               nan | (70, 25) |     0.0153   |    0.01682 | off the road almost immediately to left                                                                         | nan                                  |
| JohnNewAll, JohnFailCorrPostBridgePreBridgePlus | mse         |                    1 |                    0 |       0     |         0 |            0.5 |              1   | False           |                  nan   | Nvidia     | True   | False         |               nan | (70, 25) |     0.0048   |    0.01017 | pretty decent but got tricked by off ramp after bridge, hugged right edge before bridge and left rail of bridge | nan                                  |
| JohnNewAll, JohnFailCorrPostBridgePreBridgePlus | mse         |                    1 |                    1 |       0     |         0 |            0.5 |              0.8 | False           |                  nan   | Nvidia     | True   | False         |               nan | (70, 25) |     0.0141   |    0.02117 | off road to right before first big curve                                                                        | nan                                  |
| JohnNewAll, JohnFailCorrPostBridgePreBridgePlus | mse         |                    1 |                    1 |       0.001 |         0 |            0.5 |              1   | False           |                  nan   | Nvidia     | True   | False         |               nan | (70, 25) |     0.0086   |    0.00879 | off road to water before bridge                                                                                 | nan                                  |
| JohnNewAll, JohnFailCorrPostBridgePreBridgePlus | mse         |                    1 |                    1 |       0     |         0 |            0.5 |              1   | True            |                    0.1 | Nvidia     | True   | True          |               nan | (70, 25) |     0.0047   |    0.00438 | off road to water before bridge                                                                                 | nan                                  |
| JohnNewAll, JohnFailCorrPostBridgePreBridgePlus | mae         |                    1 |                    1 |       0     |         0 |            0.5 |              1   | True            |                    0.1 | Nvidia     | True   | True          |               nan | (70, 25) |     0.0047   |    0.00438 | off road to water before bridge                                                                                 | nan                                  |
| JohnNewAll, JohnFailCorrPostBridgePreBridgePlus | mae         |                    1 |                    1 |       0     |         0 |            0.5 |              1   | True            |                    0.1 | Nvidia     | True   | True          |               nan | (70, 25) |     0.0572   |    0.05893 | off road to water before bridge                                                                                 | nan                                  |
| JohnNewAll, JohnFailCorrPostBridgePreBridgePlus | mae         |                    1 |                    1 |       0     |         0 |            0.5 |              1   | True            |                    0.2 | Nvidia     | True   | True          |               nan | (70, 25) |     0.0653   |    0.11543 | off road to water before bridge                                                                                 | nan                                  |
| JohnNewAll, JohnFailCorrPostBridgePreBridgePlus | mse         |                    1 |                    1 |       0     |         0 |            0.5 |              1   | False           |                  nan   | Nvidia     | True   | False         |               nan | (70, 25) |   nan        |  nan       | nan                                                                                                             | nan                                  |
| ClockwiseCenter                                 | mse         |                    1 |                    0 |       0     |         0 |            0.5 |              1   | True            |                    0.2 | Nvidia     | True   | True          |               nan | (70, 25) |     0.0084   |    0.01197 | off road to water before bridge                                                                                 | nan                                  |
| ClockwiseCenter, FailureCorrection              | mse         |                    1 |                    0 |       0     |         0 |            0.5 |              1   | True            |                    0.2 | Nvidia     | True   | True          |               nan | (70, 25) |   nan        |  nan       | tricked after bridge, ran into right side of off ramp                                                           | nan                                  |
| ClockwiseCenter, FailureCorrection              | mse         |                    1 |                    0 |       0     |         0 |            0.5 |              1   | True            |                    0.3 | Nvidia     | True   | True          |               nan | (70, 25) |     0.0121   |    0.03821 | tricked after bridge, ran into right side of off ramp                                                           | nan                                  |
| ClockwiseCenter, FailureCorrection              | mse         |                    1 |                    0 |       0     |         0 |            0.5 |              1   | True            |                    0.5 | Nvidia     | True   | True          |               nan | (70, 25) |     0.0486   |    0.11716 | off road to water before bridge                                                                                 | nan                                  |
| ClockwiseCenter, FailureCorrection2             | mse         |                    1 |                    0 |       0     |         0 |            0.5 |              1   | True            |                    0.2 | Nvidia     | True   | True          |               nan | (70, 25) |     0.0076   |    0.01058 | tricked after bridge, ran into left side of off ramp                                                            | nan                                  |
| ClockwiseCenter, FailureCorrection2.1           | mse         |                    1 |                    0 |       0     |         0 |            0.5 |              1   | True            |                    0.2 | Nvidia     | True   | True          |               nan | (70, 25) |     0.0128   |    0.01436 | tricked after bridge, drove into off ramp                                                                       | nan                                  |
| ClockwiseCenter, FailureCorrection2.2           | mse         |                    1 |                    0 |       0     |         0 |            0.5 |              1   | True            |                    0.2 | Nvidia     | True   | True          |               nan | (70, 25) |     0.0091   |    0.01158 | tricked after bridge, drove into off ramp worse!                                                                | nan                                  |
| sample_data                                     | mse         |                    1 |                    0 |       0     |         0 |            0.5 |              1   | True            |                    0.2 | Nvidia     | True   | True          |               nan | (60, 25) |     0.0276   |    0.01369 | made it to second turn in post-bridge s-curve then into water to left                                           | nan                                  |
| ClockwiseCenter                                 | mse         |                    1 |                    0 |       0     |         0 |            0.5 |              1   | True            |                    0.2 | Nvidia     | True   | True          |               nan | (60, 25) |     0.0089   |    0.00644 | tricked after bridge, drove into off ramp                                                                       | nan                                  |
| ClockwiseCenter, CounterClockwiseCenter         | mse         |                    1 |                    0 |       0     |         0 |            0.5 |              1   | True            |                    0.3 | Nvidia     | True   | True          |               nan | (60, 25) |     0.0208   |    0.01516 | tricked after bridge, almost made it, off road left of off ramp                                                 | nan                                  |
| ClockwiseCenter, CounterClockwiseCenter         | mse         |                    1 |                    0 |       0     |         0 |            0.5 |              1   | True            |                    0.4 | Nvidia     | True   | True          |               nan | (60, 25) |     0.0442   |    0.02924 | Success!                                                                                                        | nan                                  |
#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I experimented with udacity sample data as well as personally captured center driving in both directions, recovery driving in both directions, and specific recovery driving of identified failure modes.  In the end, I chose to use only Centered driving in both directions around the track along with left/right camera based corrections.

At one point I was using all of my captured data and creating more without success specifically after the bridge.  Eventually, through experimentation and some helpful Slack colleagues I realized that I needed to modify my cropping dimensions to include more of the image.  After cropping got me past my problem spot, I decided to start from the most basic dataset and build it up gradually in favor of simplicity and generality.

For details about how I created the training data, see the next section. 

### Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to choose an appropriate existing architecture based on similarity of task.

As stated above, I started with an attempt using LeNet5 but ultimately used a convolution neural network model similar to the [Nvidia](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) architecture. Given that it had specificaly been used for autonomous driving, it seemed more than appropriate, the primary risk being excessive model complexity for the simple simulator environment.

In order to gauge how well the model was working, I split my image and steering angle data into an 80%/20% training and validation set. At first, with no regularization strategies deployed aside from Early Stopping (which indeed prevented an excessive delta between training and validation losses) I observed a suboptimally large difference between the training and validation losses indicating overfitting.

To combat the overfitting, I modified the model to use dropout and L2 regularization in addition to early stopping.  Ultimately, I found that 50% dropout on the fully connected layers had the most positive impact in my experiments. (see table above for the experiment log)

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track; in particular, the water just before the bridge and the dirt off ramp following the bridge tended to pull my car off the main road. This happened with almost all permutations of my dataset including the Udacity sample data. To improve the driving behavior in these cases, I did two things:

1. I changed my top crop-out pixels from 70 to 60. This seemed to be the key that made it possible to make progress beyond the dirt off ramp; it made the Udacity sample data progress beyond the dirt off ramp and fail elsewhere. My previous cropping must have been masking away road that would otherwise allow the car to infer a continued turn on the paved road.
2. After the cropping breakthrough, I went back to my self-made dataset but found that I needed to increase the left/right camera steering correction from 0.2 to 0.4. I did this in 0.1 steps finding that 0.4 finally made it overtake the dirt off ramp and actually, to my surprise, make it through the entire track.

Though there is opportunity to make the driving smoother and generalize it further to track 2, the vehicle is able to drive autonomously around the track 1 without leaving the road. The limits of time have forced me to move on to the next lesson and leave this for another day.

#### 2. Final Model Architecture

The final model architecture (model.py lines 84-140) is a convolution neural network with the following layers and layer sizes:

| layer      | description                                                | out_shape   |
|:-----------|:-----------------------------------------------------------|:------------|
|            | Input image                                                | 160x320x3   |
| Lambda     | Normalization                                              | 160x320x3   |
| Cropping2D | Image cropping removing 60 pixels from top, 25 from bottom | 75x320x3    |
| Conv2D     | 5x5 kernel, 2x2 stride, valid padding, relu activation     | 36x158x24   |
| Conv2D     | 5x5 kernel, 2x2 stride, valid padding, relu activation     | 16x77x36    |
| Conv2D     | 5x5 kernel, 2x2 stride, valid padding, relu activation     | 6x37x48     |
| Conv2D     | 3x3 kernel, 1x1 stride, valid padding, relu activation     | 4x35x64     |
| Conv2D     | 3x3 kernel, 1x1 stride, valid padding, relu activation     | 2x33x64     |
| Dropout    | 50% keep probability                                       | 2x33x64     |
| Flatten    | 2x33x64 = 4224                                             | 4224        |
| Dense      | fully connected, linear activation                         | 100         |
| Dropout    | 50% keep probability                                       | 100         |
| Dense      | fully connected, linear activation                         | 50          |
| Dropout    | 50% keep probability                                       | 50          |
| Dense      | fully connected, linear activation                         | 10          |
| Dropout    | 50% keep probability                                       | 10          |
| Dense      | fully connected, linear activation                         | 1           |
|            | Output steering angle (real number)                        | 1           |


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. I also recorded two laps of center lane driving in the other direction.  Here is an example image of center lane driving:

![good driving center][image1]

I also used the corresponding left and right camera images to impose a steering correction towards 'center' on the good driving data.  Here are the left and right images corresponding to the center lane driving above:

![good driving left][image2]

![good driving right][image3]

That's all I used for the final model!

Nevertheless, I did record the vehicle recovering from the left side and right sides of the road back to center to experiment with a data set which would explicitly incentivize the vehicle to come away from the edge of the road.  These images show what a recovery looks like (it's admittedly subtle):

![recovery 1][image4]

![recovery 2][image5]

![recovery 3][image6]


The good driving data in both directions sum to 8026 time steps. The center camera data and corresponding steering correction data from both left and right camera images total 3*8026 = 24078 example (image, steering angle) pairs.

I had Keras randomly shuffle the data set and put 20% of the data into a validation set before each training run.

The validation set helped determine if the model was over or under fitting.  Under fitting would manifest as very similar training and validation loss (particularly in the case of many epochs) while over fitting would manifest as a much lower training loss than validation loss.  The Nvidia model was complex enough to prevent underfitting.  To prevent overfitting I used various forms of regularization (mostly dropout as discussed above) and automatically stopped when the validation loss hadn't improved for 5 epochs in a row.  I found that that was actually what I tended to do when manually fiddling with num_epochs.  I then only saved the model at the best validation loss epoch.  I used an adam optimizer so that manually training the learning rate wasn't necessary.
