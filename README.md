# :open_file_folder: Digits-Images-Classification


This project is a Kaggle project: https://www.kaggle.com/c/dsti-s20-ann/leaderboard .

It aims at classifying a very noisy version of the MNIST dataset.

With the proposed network and the tuning done I achieved an accuracy of 96.88 %


<p align="center">
  <img src="https://miro.medium.com/max/2198/1*s9ZgQMdAbuUYKdoeCgrTZg.gif"/>
</p>

# Classify digits for a more complicated version of MNIST dataset

## Synopsis

In the field of Computer Vision, image classification is a well know and powerful method that has been enhanced in order to produce some remarkable achievements. This project challenges in attempting the classification of digits in the range of 0 to 9 through the usage of a Convolutional Neural Network. We have been provided with a nosy dataset already divided into training set and testining set. Following the CNN process, we have then subdivide the training set into train set and validation set in order to tune our model and execute the predictions. The best model we were able to build scored us an accuracy of 0.9688 %.

## Tools

The codes have been scripted into Python and due to some technical issues, we were constrained to work on Visual Studio Code. The mostly used tools are the Keras framework for building the Convulational Neural Network, numpy for preprocessing and pandas for creating the final dataframe containing the predictions. All predictions were written to csv files.

## Set up of the working environment

Make sure that you have install these main modules:

- Numpy in order to transform, manipulate and process the data
- Matplotlib in order to plot
- Scikit-learn's train-test_split in order to divide the train set into a train and validation set
- Keras in order to build the model and make the preprocessing
- Pandas in order to read in the csv file and generate the final sample

_see [code](https://github.com/IsmaelMekene/Digits-Classification-/blob/main/ANN_Digits_Classification.ipynb)_

## Data loading

Downloaded from a private Kaggle challenge created by Dr Nadiya Shvai for S20. The data is made of 12000 observations in training set and 50000 observations in testing set, respectively with 785 and 784 rows. Datasets have firstly been downloaded from Kaggle and the loaded into the PYNB as follow: [code](https://github.com/IsmaelMekene/Digits-Classification-/blob/main/ANN_Digits_Classification.ipynb)

## Data Preprocessing

The respective shapes of the loaded train and test sets are as follow: (12000,785) and (50000,784) We have done the preprocessig part in multiple parts:

- Dividing the train set into train and validation set (with respect to the 0.8:0.2 ratio)

- Dividing the new train set into xtrain (normalised) and ytrain (the last 785th column, consisted of the class of digits); applied the same steps to the validation set. We then have four created sets from train and validation sets added to the known test sets that have been converted into arrays.

- Reshaping the xtrain set into the raw pixel form of (28x28), this allows to observ the noise.

- Building an autoencoder in order to clear the noise up.

_see [code](https://github.com/IsmaelMekene/Digits-Classification-/blob/main/ANN_Digits_Classification.ipynb)_

<p align="center">
  <img src="https://github.com/IsmaelMekene/Metaheuristics--Stochastic-Optimization/blob/main/images/digitviz.svg"/>
</p>


Although we have built the autoencoder, tuning its parameters has been very difficult and it is still not the aimed result as we can see on the compared images. We have managed to adapt it to the model and keep the process up.



Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_7 (Conv2D)            (None, 28, 28, 16)        160       
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 28, 28, 32)        4640      
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 28, 28, 32)        9248      
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 28, 28, 16)        4624      
_________________________________________________________________
dense (Dense)                (None, 28, 28, 1)         17        
=================================================================
Total params: 18,689
Trainable params: 18,689
Non-trainable params: 0
_________________________________________________________________
Epoch 1/5
240/240 [==============================] - 43s 177ms/step - loss: 0.0174 - val_loss: 0.4991
Epoch 2/5
240/240 [==============================] - 51s 213ms/step - loss: 0.0041 - val_loss: 0.3678
Epoch 3/5
240/240 [==============================] - 51s 212ms/step - loss: 0.0036 - val_loss: 0.2714
Epoch 4/5
240/240 [==============================] - 50s 209ms/step - loss: 0.0030 - val_loss: 0.1713
Epoch 5/5
240/240 [==============================] - 51s 214ms/step - loss: 0.0025 - val_loss: 0.0622
Out[60]:
<tensorflow.python.keras.callbacks.History at 0x25700d72048>

## Comparing both sets of images

<p align="center">
  <img src="https://github.com/IsmaelMekene/Metaheuristics--Stochastic-Optimization/blob/main/images/digitboth.svg"/>
</p>

## Define Model Architecture

The architecture of our model is constisted of three fully connected hidden layers. We use Rectified Linear Units(ReLUs) f(x) = max(0, x) as the activation function (or nonlinearity) followed by each hidden layer. This non-saturating nonlinearity has been proven to be faster to train neural networks than the saturating ones which the gradients are often approaching zero in the limiting cases. When it comes to the larger datasets, learning faster would have a great impact on the performance of the model. The final layer is a 10-way softmax which produces a probability distribution over the 10 class labels. We use a cross entropy loss for the optimization of the neural network, which minimizes the negative log-probability of the correct label under the predicted probability distribution over the training examples. Although multiple model architectures were defined during the experimentation process, this selected model performed way better than the other ones and it reached a final accuracy of 96.63 percent on evaluation on the validation set. Other refinements were made before arriving at the current model which has the following architecture.

_see [code](https://github.com/IsmaelMekene/Digits-Classification-/blob/main/ANN_Digits_Classification.ipynb)_

Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_9 (Conv2D)            (None, 26, 26, 28)        280       
_________________________________________________________________
batch_normalization_6 (Batch (None, 26, 26, 28)        112       
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 24, 24, 28)        7084      
_________________________________________________________________
batch_normalization_7 (Batch (None, 24, 24, 28)        112       
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 12, 12, 28)        0         
_________________________________________________________________
dense_1 (Dense)              (None, 12, 12, 64)        1856      
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 10, 10, 48)        27696     
_________________________________________________________________
batch_normalization_8 (Batch (None, 10, 10, 48)        192       
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 8, 8, 48)          20784     
_________________________________________________________________
batch_normalization_9 (Batch (None, 8, 8, 48)          192       
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 6, 6, 64)          27712     
_________________________________________________________________
batch_normalization_10 (Batc (None, 6, 6, 64)          256       
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 4, 4, 64)          36928     
_________________________________________________________________
batch_normalization_11 (Batc (None, 4, 4, 64)          256       
_________________________________________________________________
global_average_pooling2d (Gl (None, 64)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 256)               16640     
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                2570      
=================================================================
Total params: 142,670
Trainable params: 142,110
Non-trainable params: 560
_________________________________________________________________

## Training the model

In order to tyrain the model, the hyperparameters have respectively been set to:

hyperparameters | value |
--- | --- |
batch_size | 10 |
epochs | 30 |
learningrate | 0.001 |
decay_rate | 0.001/30 |



batch_size = 10
epochs = 30
learningrate=0.001
decay_rate=learningrate/epochs
