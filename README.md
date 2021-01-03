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
