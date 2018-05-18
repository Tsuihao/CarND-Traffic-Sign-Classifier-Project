import numpy as np
import tensorflow as tf
from skimage.transform import rotate
from sklearn.utils import shuffle
import cv2
import random

class DataAugmentation:
    def __init__(self):
        self.img = tf.placeholder(tf.uint8, [32, 32, 3])
        self.rand_brightness = tf.image.random_brightness(self.img, 0.5) # tf function
        self.rand_contrast = tf.image.random_contrast(self.img, 0.5, 1.5) # tf function
        self.rand_saturation = tf.image.random_saturation(self.img, 0.5, 2) # tf function
        self.rand_hue = tf.image.random_hue(self.img, 0.05) # tf function
        
    def image_process(self, sess, image, ):
        operation_num = random.randint(1, 5)
        operation_seq = [True]*operation_num + [False]*(5 - operation_num) # e.g True True False False False
        operation_seq = shuffle(operation_seq) #e.g. False True False True False
        
        # op-1 cropping
        if operation_seq[0]: image = self.random_crop(image)
        
        # op-2 brightness
        if operation_seq[1]: sess.run(self.rand_brightness, feed_dict={self.img: image}) 
            
        # op-3 contrast
        if operation_seq[2]: sess.run(self.rand_contrast, feed_dict={self.img: image})
            
        # op-4 saturation
        if operation_seq[3]: sess.run(self.rand_saturation, feed_dict={self.img: image})
            
        # op-5 hue
        if operation_seq[4]: sess.run(self.rand_hue, feed_dict={self.img: image})
        
        return image
    
    def random_crop(self, image):
        crop_size = random.randint(24,32)
        offset = 32 - crop_size # 32 is the default
        x1 = random.randint(0, offset)
        x2 = x1 + crop_size
        y1 = random.randint(0, offset)
        y2 = y1 + crop_size
        img_crop = image[y1:y2, x1:x2, :] #?
        img_resize = cv2.resize(img_crop, (32,32))
        return img_resize
    
    def random_rotate(self, image):
        rand_angle = random.randint(-5, 5)
        rot = rotate(image, rand_angle)*255
        rot = rot.astype(np.uint8)
        return rot
    
                                                                                                                
        
        