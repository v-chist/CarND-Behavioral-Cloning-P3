# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[network]: ./images/cnn-architecture.png "Neural network"
[image]: ./images/image.jpg "Image"
[image_flipped]: ./images/image_flipped.jpg "Image_flipped"
[center]: ./images/center.jpg "Center driving"
[original]: ./images/original.jpg "Original image"
[cropped]: ./images/cropped.jpg "Cropped image"





## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes. The model includes RELU activation to introduce nonlinearity. The data is normalized in the model using a Keras lambda layer.

    model = Sequential()
    
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
    
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    
    model.add(Convolution2D(64,3,3,activation='relu'))
    
    model.add(Convolution2D(64,3,3,activation='relu'))
    
    model.add(Flatten())
    
    model.add(Dense(100))
    
    model.add(Dense(50))
    
    model.add(Dense(10))
    
    model.add(Dense(1))
 

#### 2. Attempts to reduce overfitting in the model


The model was trained and validated on different data sets to ensure that the model was not overfitting:

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)


 The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used center lane driving. Recovering from the left and right sides of the road was not necessary since i used data from side cameras to increase the data set.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to take some existing architecture and modify it if needed.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (20% for validation set).

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it is widely used for image classification task and can be a good starting point for prediction. I used 5 epochs to train model. The model was not overfitting - accuracy on validation set was decreasing from 1st to 5th epoch. This model showed pretty good result and was able to take the car to the bridge.

Second approach was a more complicated model - i used the model architechture that was used by NVIDIA team for training self driving car. The architechture was presented in the lectures. See the description of the layers below:

 

To combat the overfitting, I reduced the number of epochs (used just 2 epochs)

The car was driving really well and was able to drive the whole track autonomously.

#### 2. Final Model Architecture

The final model architecture (see model.py) consisted of the following convolution neural network:

    model = Sequential()
    
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
    
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    
    model.add(Convolution2D(64,3,3,activation='relu'))
    
    model.add(Convolution2D(64,3,3,activation='relu'))
    
    model.add(Flatten())
    
    model.add(Dense(100))
    
    model.add(Dense(50))
    
    model.add(Dense(10))
    
    model.add(Dense(1))


Here is the description of the network from Nvidia:

>  
> The first layer of the network performs image normalization. The normalizer is hard-coded and is not adjusted in the learning process. Performing normalization in the network allows the normalization scheme to be altered with the network architecture, and to be accelerated via GPU processing.
> 
> The convolutional layers are designed to perform feature extraction, and are chosen empirically through a series of experiments that vary layer configurations. We then use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers.
> 
> We follow the five convolutional layers with three fully connected layers, leading to a final output control value which is the inverse-turning-radius. The fully connected layers are designed to function as a controller for steering, but we noted that by training the system end-to-end, it is not possible to make a clean break between which parts of the network function primarily as feature extractor, and which serve as controller.
> .
> 

Visualization of the network:


![alt text][network]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

After the collection process, I had 4150 images.

To augment the data sat, I also flipped images and angles thinking that this would help to avoid overtraining medel to turn only to the left. For example, here is an image that has then been flipped:

![alt text][image]
![alt text][image_flipped]

This gave me 8300 images.

I also used side camera images with 0.2 correction to steering. This allows model to train how to get back to the center of the road and helped a lot to improve the model behavior.

This gave me the final number of 24900 images. I trained the model on random 80% () of the images and used 20% for validation set.

I preprocessed this data by cropping the images from top and bottom. Here is an example of cropping:

![alt text][original]
![alt text][cropped]


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by the fact that validation loss was not increasing. I used an adam optimizer so that manually training the learning rate wasn't necessary.

    Training...
    Train on 19920 samples, validate on 4980 samples
    Epoch 1/2
    19920/19920 [==============================] - 656s - loss: 0.0098 - val_loss: 0.0219
    Epoch 2/2
    19920/19920 [==============================] - 629s - loss: 0.0052 - val_loss: 0.0172
    Model saved
