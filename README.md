# FruterIA 🍎🍊🍌

You can find the final report [here](report/MCPR2020_paper.pdf).

## 🍉 Abstract

Payment of fruits or vegetables in retail stores normally require them to be manually identified. This paper presents an image clas- sification method, based on lightweight Convolutional Neural Networks (CNN), with the goal of speeding up the checkout process in stores. A new dataset of images is introduced that considers three classes of fruits, inside or without plastic bags. In order to increase the classification accuracy, different input features are added into the CNN architecture. Such inputs are, a single RGB color, the RGB histogram, and the RGB centroid obtained from K-means clustering. The results show an overall 95% classification accuracy for fruits with no plastic bag, and 93% for fruits in a plastic bag.

## 🍇 Setup

The packages needed are the following:
* tensorflow >= 2.0
* numpy
* OpenCV2
* matplotlib
* pandas
* seaborn
* fire
* keract
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
