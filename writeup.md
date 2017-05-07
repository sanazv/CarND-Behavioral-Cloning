Files included:
For this submission I am including the following files as per project rubric requirements:
* model.py -- python script which reads and pre-processes data and creates and trains the model.
* drive.py -- the script provided by Udacity that drives the car - I have changed the speed parameter to generate two video files.
* model.h5 -- weights of the trained network
* video.mp4 -- a video recording of my vehicle driving autonomously for a little over one lap (until second visit of the cobblestone bridge) at speed 9mph
* video_fast.mp4 -- a video recording of my vehicle driving autonomously for a little over one lap (until second visit of the cobblestone bridge) at speed higher speed of 20 mph

[//]: # (Image References)
[image1]: ./plots/left_view.png "Left"
[image2]: ./plots/center_view.png "Center"
[image3]: ./plots/right_view.png "Right"

[image4]: ./plots/original_steering_angles.png "Steering Angles"


[image5]: ./plots/all_steering_angles_post_correction.png "From three cameras"

[image6]: ./plots/all_steering_angles_peakiness_removed.png "Reduce dominant angles"


[image7]: ./plots/before_flip.png "Before"
[image8]: ./plots/after_flip.png "After"
[image9]: ./plots/final_angle_dist.png "Final Distribution"
[image10]: ./plots/nn_arch.png "Architecture"


## Introduction: 
In this project the goal was to design and train a network network to successfully drive a car around the simulated tracks without ever touching any of the drivable segments of the road.
In this document I will explain how I went about pre-processing the dataset, setting up the architecture and training the model. The result of the autonomous drive is provided in two video files.
I generated two videos with the same model weights. The first one with the default driving speed of 9 mph (in drive.py) and the second one with increased speed of 20mph. In both cases the autonomous car drives the track successfully without touching any of the side lines at any point throughout the track. I kept the videos a bit longer than one lap (stopped at second time passing the cobblestone bridge) to make sure one full lap is included.
I use a GeForce GTX 1060 6GB GPU for this project and did not need to use a generator. Also I have used Keras version 2.0.3 for this project.


## Data Pre-Processing:
Since I am not very good at playing video games, I figured the data I would generate to train the network might not set a very good example for the network. I have asked friends to help me and drive the car on the track. However I also decided to invest in augmenting the provided data set as much as possible to achieve the goal of a successful drive. 
In this section I will explain how I enhanced the provided data set and prepared the training set. 
The dataset provided includes images from 3 camera angles (Center, Right and Left), with 8036 images for each camera for a total of 24,108 images and steering angles. Each image is 160x320 in 3 color channels.  Below I show examples of these view points:

![alt text][image1]
![alt text][image2]
![alt text][image3]



Looking at the distribution of the steering angles (as shown below) it can be seen that majority of the data points are from instances where the steering angle (from point of view of the central camera) is 0. Having such non-uniform distribution of steering angles can cause a bias in the way the network learns from the data. 


![alt text][image4]


In order to avoid this bias I will remove a fraction of data (angles and their corresponding images) where the steering angle is 0 to have a more uniform distribution of various examples in the training set. I choose to keep 10% of all the dominant steering angles as well as all the non dominant steering angles.
However doing this would reduce the number of training samples which is not ideal. In order to counter balance this, I include the data from left and right cameras. The provided steering angle file only contains steering angle values from point of view of the central camera. To mimic the same for the side cameras I add a correction factor to the steering angles when side camera images are taken in. 
The value of this correction factor is +0.1 and -0.1 for let and right cameras respectively. 
I assume that the width of the car is ~ 2m. So the distance between any of the side cameras to the central camera is ~1m. If the steering angle is taking effect is about 10m in front of the car, the correction for the side cameras wih small angle approximation would be: $\alpha = \tan(\alpha) = \frac{1}{10}=0.1$. I use this correction factor to correct the central steering angle for both of the side cameras.

The plot below shows the distribution of steering angles after side cameras' data is included. Now it can be seen that a large fraction of data has steering angles equal to = [0, -0.1, +0.1]. 

![alt text][image5]

After this stage, I remove the peskiness of this distribution by discarding a large fraction of data with these particular steering angle values. At this stage I have a total of 12,333 images from all three cameras, which is still about half of the orifinal dataset. The figure below shows the resulting distribution.

![alt text][image6]



At this stage, in order to increase the size of training set and provide more general dataset to the network, I augment the dataset by flipping each image horizontally (with negative of the original steering angle) and adding them to the original training set. By taking this step I double the number of images in the training set (now at 24666), which proves to be helpful in how the network learns to generalise.
The figures below show an example of the original and flipped image side by side.

![alt text][image7]
![alt text][image8]


The resulting steering angle distribution is shown here: 

![alt text][image9]


which can be seen to be more symmetric and uniform compared to the original distribution.
As the final step I exaggerate the steering angles by 20% to encourage the network to made stronger decisions at turns. 
With all the steps taken above I ensure that the network has a balanced and well representative training data to learn from.


## Model Architecture and Training Strategy:
As for the model I initially experimented with custom architectures and added layer after later to see how the car would perform. At the end, I decided to follow the architecture described in this paper [End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316). At first I normalise the pixel values to 1 and then remove the mean from all pixels. I then crop each image such that the top 50 (sky etc) and bottom 25 (hood of the car) pixels are removed. The resulting images are now 85x320 rather than 160x320. The architecture used in the Nvidia paper has 5 back to back convolution layers, before flatting and going through fully connected dense layers. The first 3 convolutional layers have kernel size of 5x5 with stride of 2 and the last 2 convolutional layers have kernel size of 3x3 with stride of 1. I have added a dropout later with 50% dropout at the end of convolution layers before flattening to reduce over fitting. At each convolution layer I use the ELU (Exponential Linear Unit) to introduce non-linearities and avoid diminishing gradient as well as added speed benefit over ReLU activation layers.
the data is split to training and validation set (70%-30%) and I set shuffling to be true, so the frames will be fed to the network in random order and not in the order of drive, which enables the network to generalise better whenever faced with a new image.
I used a Adam optimiser with learning rate (0.0001) after trying the default value of ?? and not getting as nice results. 
Also using batch size of 64 and number of epochs 10. At first I used larger number of epochs but even though the training loss would decrease, the validation loss was plateauing suggesting that with more epochs the network starts to overfit. So I stop the number of epoch to 10, so that I have steady decrease of validation loss and start to plateau.
The graph below shows the summary of the network architecture: 
 
![alt text][image10]



## Summary:

Success- can be seen from the video


