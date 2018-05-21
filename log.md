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


**21.May.2018**
* Review finding from: [Here](https://review.udacity.com/?utm_medium=email&utm_campaign=ret_000_auto_ndxxx_submission-reviewed&utm_source=blueshift&utm_content=reviewsapp-submission-reviewed&bsft_clkid=631f0f99-967c-4000-b3ed-a9b6e0c62599&bsft_uid=1e07ff6b-26c2-40f5-8927-a5800725c305&bsft_mid=766e8c83-0199-44a4-a16e-54a264908e9f&bsft_eid=6f154690-7543-4582-9be7-e397af208dbd&bsft_txnid=e1b6f06f-541b-42ca-a9f3-f833b7e4023e#!/reviews/1230868)
* Add Tensorboard images
* Add desciption to new test images.

---
**Reference**:

**Data augmentation**: https://github.com/AlphaLFC/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb

