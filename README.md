## Project: Traffic Sign Recognition
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
Traffic sign recognition

Dataset: [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)


The Project
---
The goals / steps of this project are the following:
* Preprocess the dataset: convert to grayscale, normalization, augmentation. Check: **[preprocess](https://github.com/Tsuihao/CarND-Traffic-Sign-Classifier-Project/blob/master/src/preprocess.py)** and **[dataaugmentation](https://github.com/Tsuihao/CarND-Traffic-Sign-Classifier-Project/blob/master/src/dataaugmentation.py)**
* Design, train and test a model architecture. Check: **[cnnarchitecture](https://github.com/Tsuihao/CarND-Traffic-Sign-Classifier-Project/blob/master/src/cnnarchitect.py)**
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report. Check: **[writeup](https://github.com/Tsuihao/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup.md)**


### Result and Log

Check here: **[Result](https://github.com/Tsuihao/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup.md)**

Check here: **[Log](https://github.com/Tsuihao/CarND-Traffic-Sign-Classifier-Project/blob/master/log.md)**

---

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the **original** project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```