**13.May.2018**

**preprocess:** convert to grayscale

**architecture:** [cnn_arc_1](https://github.com/Tsuihao/CarND-Traffic-Sign-Classifier-Project/blob/master/src/cnnarchitect.py)

**test accuracy:** 94.9

**Hyperparameters:**

- learning rate: 0.0009
- batch size: 128
- epochs: 60

TODO:
* Add data augmentation
* use ```from sklearn.model_selection import GridSearchCV``` to find out the best hyperparameters. [Reference](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

Doubts:
* First normalization or convert into gray scale? 
        If normalization -> grayscale the accuracy is: 86
        If grayscale -> normalization the accuracy is: 94.5+
---
**17.May.2018**

* Try with the different normalization method: [cv2_gray_n_normalize](https://github.com/Tsuihao/CarND-Traffic-Sign-Classifier-Project/blob/master/src/preprocess.py)<br>
 **result:** No difference!

* Try with data augmentation: [dataaugmentation](https://github.com/Tsuihao/CarND-Traffic-Sign-Classifier-Project/blob/master/src/dataaugmentation.py)<br>
**result:** test accuracy from 94.9% ->96.2% (with same configuration, the only difference is batch size 128 -> 256 due to more data)
        
        
---
**19.May.2018**

* Check the padding VALID and SAME in tesorflow

---

**20.May.2018**
* Refoctory the code into encapsulated class
* Clean up the user interface
* Add architect [arc_2](https://github.com/Tsuihao/CarND-Traffic-Sign-Classifier-Project/blob/master/src/cnnarchitect.py), search 'arc_2'
* Hyperparameter tuning

* **TODO**: Gridsearch

---
**Reference**:

**Data augmentation**: https://github.com/AlphaLFC/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb

