# **Traffic Sign Recognition** 

## Writeup
---
[all_new_test]: ./images/test8.png 
[training_samples_distribution]: ./images/dataset_original.png 
[augmented_distribution]: ./images/dataset_augmented.png 
[training_samples_rgb]: ./images/training_samples_rgb.png 
[training_samples_grayscale]: ./images/training_samples_grayscale.png 
[top5]: ./images/top5.png 
[each_class]: ./images/each_class.png 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

---
## Definition of Done (DoD)/ Requirements: [rubric points](https://review.udacity.com/#!/rubrics/481/view) 


### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Link to my [project code](https://github.com/Tsuihao/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the **numpy** library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
```python
import numpy as np
n_train = len(y_train)
```
* The size of the validation set is ?
```python
n_validation = len(y_valid)
```
* The size of test set is ?
```python
n_test = len(y_test)
```
* The shape of a traffic sign image is ?
```python
image_shape = X_train.shape[1:3]
```
* The number of unique classes/labels in the data set is ?
```python
n_classes = len(np.unique(y_train))
```

Number of training examples = 34799 <br>
Number of testing examples = 12630 <br>
Image data shape = (32, 32)<br>
Number of classes = 43<br>

#### 2. Include an exploratory visualization of the dataset.

The distribution of dataset 

![alt text][training_samples_distribution]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

**[Image preprocessing]**

reference: 
* [Image Processing for Deep Learning](https://dziganto.github.io/deep%20learning/image%20processing/python/Image-Processing-for-Deep-Learning/) 1. rgb to gray 2. downsample 3. normalization
* [Image Data Pre-Processing for Neural Networks](https://becominghuman.ai/image-data-pre-processing-for-neural-networks-498289068258) 1.uniform aspect ration 2. image scaling 3. normalization 4. rgb to gray 5. data augmentation

Step 1: Convert rgb to grayscale based on this [reference](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) and this [formula](https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale). The reason of converting the rgb to grayscale is becasue the quality of the provided rgb images is low.
By doing this, the classifier will focuse on the contour of the traffic signs. However, in real life scenario, the color infomation of traffic sign is vital.

RGB
![alt text][training_samples_rgb]

Grayscale
![alt text][training_samples_grayscale]

Step 2: Normalized the data

[Naive normalization](https://github.com/Tsuihao/CarND-Traffic-Sign-Classifier-Project/blob/master/src/preprocess.py)

Step 3: Argumented the data

[Data augmentation](https://github.com/Tsuihao/CarND-Traffic-Sign-Classifier-Project/blob/master/src/dataaugmentation.py).

As can be seen in the folowing figure, the original dataset distribution is not balnaced. Therefore, I augmented the data to have equal distribution.

![alt text][augmented_distribution]


**[Build up CNN architect]**

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   				    | 
| Conv_1 3x3         	| 1x1 stride, same padding, outputs 32x32x30 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, same padding, outputs 16x16x30	|
| Conv_2 3x3	        | 1x1 stride, same padding, outputs 16x16x60    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, same padding, outputs 8x8x60      |
| Conv_3 3x3            | 1x1 stride, same padding, outputs 8x8x120     |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, same padding, outputs 4x4x120     |
| flatten               | outputs:1960                                  |
| FC_4                  | outputs: 840                                  | 
| FC_5          		| outputs: 256        							|
| FC_6                  | outputs: 43                                   |
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

learing_rate: 0.0003 <br>
batch_size: 256 <br>
epochs: 60<br>
optimizer: Adam<br>

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ? 100.00%
* validation set accuracy of ? 99.791%
* test set accuracy of ? 95.90%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? <br>
Check: [arc_1](https://github.com/Tsuihao/CarND-Traffic-Sign-Classifier-Project/blob/master/src/cnnarchitect.py) and search: 'arc_1'

* What were some problems with the initial architecture?<br>
The initial conv_layer output feature maps is too less. I increased the output number of feature maps (which also increase the training time), but the accuracy is improved.


* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.<br>
Both arc_1 and arc_2 are based on the LetNet structure, arc_2 just increases the number of feature maps. Both architectures are not over-fitting or under-fitting.

* Which parameters were tuned? How were they adjusted and why?<br>
learning_rate is tuned in range of 0.001 - 0.0001. Just experiment a bit and check the result. However, there is no significant impact in this range.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?<br>
The structure is based on the LetNet but with smaller size of convolution kernel size.
The reason is that the reference from the [VGG net](https://arxiv.org/abs/1409.1556).


If a well known architecture was chosen:
* What architecture was chosen?<br>
Modified LetNet.

* Why did you believe it would be relevant to the traffic sign application?<br>
Since the traffic sign is not a complex task to distingush, therefore, I think the LetNet can perform well. And the result proves my original thought as well.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?<br>
training: 100.0% <br>
validation: 99.791% >br?
test: 95.9%
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 8 German traffic signs that I found on the web:

![alt text][all_new_test]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric) & <br> 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts).

I will answer question 2 and 3 together. As you can see in the following first figure, the model predicts 100% in all the new test images. (Please note that each row is the new test data and from the second column on is the softmax output). As can be seen, the model solves these new test images perfectly (the first prediction is 100%, the rests are all 0%).<br>
The following second image shows the validation result of the model on each class. The majority of classes are with high accuracy, however, class 27 shows only 50% accuracy. And we can go back to see the original dataset distribution. It is clear that class 27 has very less original data and even I augmented it but the augmentation techniques also based on the original images. Therefore, class 27 can be foreseen to have bad classification quality. In addition, the original dataset's image quality will heavily affect the prediction.

![alt text][top5]


![alt text][each_class]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


