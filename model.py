
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2
import skimage.io
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout, Activation, Conv2D, Reshape
from keras_tqdm import TQDMNotebookCallback
from keras.optimizers import Adam
from skimage.transform import resize
print(keras.__version__)
# data directory
data_dir = "./examples/data/data_sample/"
img_dir = data_dir +'IMG/'

# read in the csv file with image file names and steering angle values
df = pd.read_csv(data_dir + "driving_log.csv")
df = df.replace("IMG/","", regex=True)

print("total number of frames from each camera view is: ",len(df))

#plot and save some frames
img2 = skimage.io.imread(img_dir+ df.center.values[0].replace(" ",""))
plt.imshow(img2)
plt.title('Center Camera View')
#plt.show()
plt.savefig('./plots/center_view.png')

img2 = skimage.io.imread(img_dir+ df.left.values[0].replace(" ",""))
plt.imshow(img2)
plt.title('Left Camera View')
#plt.show()
plt.savefig('./plots/left_view.png')

img2 = skimage.io.imread(img_dir+ df.right.values[0].replace(" ",""))
plt.imshow(img2)
plt.title('Right Camera View')
#plt.show()
plt.savefig('./plots/right_view.png')

# plot distribution of steering angles in the data from central camera
plt.hist(df["steering"],bins=40)
plt.xlabel('steering angle')
plt.title('steering angle from central camera')
#plt.show()
plt.savefig('./plots/original_steering_angles.png')


# read in all three view images
for camera in ['center', 'left', 'right']:
    df[camera] = df[camera].map(lambda x: img_dir+x)


# for each of the center, left and right camera angles, correct the steering angle by adding a correction factor to it.
# make a list of corrected steering angles and their corresponding images
correction_factor = [0, .1, -.1]
all_images_list = []
all_steering_angles_list = []

for camera, corr_fact in zip(['center', 'left', 'right'], correction_factor):
    tmp_images = df[camera].map(lambda x: skimage.io.imread(x.replace(" ", "")))
    all_images_list.append(np.array([v for v in tmp_images.values]))
    all_steering_angles_list.append(df['steering'] +  corr_fact)


# combine the data for all camera view points into one main array for the angles and one main for all images
all_steering_angles = np.concatenate(all_steering_angles_list)
all_images = np.concatenate(all_images_list)


print("the total number of images across all camera views is: ",all_steering_angles.shape[0])
print("shape of the image array is: ", all_images.shape)


# plot steering angles in the data from all three camera view points
plt.hist(all_steering_angles,bins=40)
plt.xlabel('steering angle')
plt.title('steering angle from all view points')
plt.savefig('./plots/all_steering_angles_post_correction.png')
#plt.show()

# remove peakiness
# take all 3 dominant steering angles and only use a small fraction of them in the training dataset
center_angle_mask = np.isclose(all_steering_angles, 0)
left_angle_mask = np.isclose(all_steering_angles, 0.1)
right_angle_mask = np.isclose(all_steering_angles, -0.1)

all_masks = center_angle_mask + right_angle_mask + left_angle_mask

bad_angle_steering = all_steering_angles[all_masks]
good_angle_steering = all_steering_angles[~all_masks]
bad_images = all_images[all_masks]
good_images = all_images[~all_masks]

divisor = 10
choices = np.random.randint(0, len(bad_angle_steering), size=int(len(bad_angle_steering)/divisor))

chosen_images = bad_images[choices]
chosen_angle_steering = bad_angle_steering[choices]

images2use = np.concatenate((good_images, chosen_images))
steering2use = np.concatenate((good_angle_steering, chosen_angle_steering))


#plot the distibution of the steering angles to use after a large portion of dominant angles are discarded
plt.hist(steering2use,bins=40)
plt.xlabel('steering angle')
plt.title('steering angle from all view points after discarding some of dominant angles')
plt.savefig('./plots/all_steering_angles_peakiness_removed.png')
#plt.show()


# flipping all images horizontally and adding them to the training set
flipped_images = np.flip(images2use, axis=2)
flipped_steering_angles = -steering2use

images_wflipped = np.vstack([images2use, flipped_images])
steering_angles_wflipped = np.hstack([steering2use, flipped_steering_angles])

# plot original and flipped image

plt.imshow(images2use[0])
plt.title('Original')
plt.savefig('./plots/before_flip.png')
#plt.show()

plt.imshow(flipped_images[0])
plt.title('Flipped')
plt.savefig('./plots/after_flip.png')
#plt.show()

# plot distribution of all steering angles including the flipped ones
plt.hist(steering_angles_wflipped,bins=40)
plt.xlabel('steering angle')
plt.title('Steering angle after number of dominant angles are reduced')
plt.savefig('./plots/final_angle_dist.png')
#plt.show()

# model architecture
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=[160, 320, 3]))
model.add(Cropping2D(cropping=((50, 25), (0, 0))))
model.add(Convolution2D(24, [5, 5], strides=(2, 2), activation='elu'))
model.add(Convolution2D(32, [5, 5], strides=(2, 2),activation='elu'))
model.add(Convolution2D(48, [5, 5], strides=(2, 2),activation='elu'))
model.add(Convolution2D(64, [3, 3], strides=(1, 1),activation='elu'))
model.add(Convolution2D(64, [3, 3], strides=(1, 1)))
model.add(Dropout(.5))
model.add(Flatten())
model.add(Dense(100,activation='elu'))
model.add(Dense(50,activation='elu'))
model.add(Dense(10,activation='elu'))
model.add(Dense(1))
model.summary()

# images being fed to train and validate the network
# exaggerate the steering angle in the training set by 20%
images = images_wflipped
angles = steering_angles_wflipped*1.2
# initialize adam optimizar with learning rate
adam = Adam(lr=0.0001)

# compile model with mean Squared Error loss function and  adam optimizar
model.compile(loss='mse', optimizer=adam)
# split training-validation set to 70-30%
# set number of ephochs and batch size and fit the model
model.fit(images, angles, validation_split=0.3, shuffle=True, epochs=10,
          batch_size=64, verbose=0)#, callbacks=[TQDMNotebookCallback()])
model.save("./model.h5")
