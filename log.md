**13.May.2018**

**preprocess:** convert to grayscale

**architecture:** [cnn_arc_1](https://github.com/Tsuihao/CarND-Traffic-Sign-Classifier-Project/blob/master/src/cnnarchitect.py)

**validation accuracy:** 96.8

**Hyperparameters:**

- learning rate: 0.0009
- batch size: 128
- epochs: 60

TODO:
* Add data augmentation
* use ```from sklearn.model_selection import GridSearchCV``` to find out the best hyperparameters. [Reference](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)