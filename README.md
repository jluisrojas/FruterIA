# FruterIA 🍎🍊🍌

You can find the final report [here](report/FruterIA-report02.pdf).

## 🍉 Abstract

The purpose of this investigation is to create a intelligent system for a Fruit Store that makes the checkout process easier. For this we propose using a Convolutional Neural Network, and during training we use techniques like transfer learning, fine tuning and data augmentation in order to make the system robust with a small dataset. To test this we introduce a new dataset of fruits images, and explore what will it take to make the system affordable and scalable. 

## 🍇 Setup

The packages needed are the following:
* tensorflow >= 2.0
* numpy
* OpenCV2
* matplotlib
* pandas
* seaborn
* fire
* gopro-python-api
* pydot

For training the model with the AOB dataset:
```
python train_AOB.py
```
To test using a GoPro, first make sure you are connected to the camera via WiFi then run:
```
python gopro_stream.py
python aob_cam.py
```

## 🍓 Training
We use the keras implementation of MobileNetV2 architecture and use transfer learning to imporve accuracy with a small dataset.

## 🍑 Dataset

For this investigation a dataset was created, contains images of apples, oranges and bananas(AOB Dataset). The images contain from one fruit to 3 frutis of the same type, and there is also photos inside of a platic bag, in total the dataset consits of 725 images for training and 342 for testing. Also there are scripts for transforming the dataset to a TFRecord to use with the Tensorflow Dataset API. The dataset directory can be found [here](src/datasets/AOBDataset/).

![alt text](https://github.com/JoseLuisRojasAranda/FruterIA/blob/master/assets/dataset.png)
