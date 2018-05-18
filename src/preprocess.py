import numpy as np
import cv2

# Y' = 0.299 R + 0.587 G + 0.114 B 
def rgb_to_gray(X_rgb):
    X_gray = []
    for sample in X_rgb:
        grayscale = np.dot(sample[...,:3], [0.299, 0.587, 0.114])
        X_gray.append(grayscale)
    
    # Turn into numpy ararry 
    X_gray = np.asarray(X_gray)
    X_gray = np.expand_dims(X_gray, axis=3)
    return X_gray


#(pixel - 128)/ 128 is a quick way to approximately normalize the data and can be used in this project.
def naive_normalization(X):
    return (X - 128)/128



def cv2_gray_n_normalize(img):
    """
    Using the embedded function in open cv to convert rgb into gray
    Using the n-mean(n)/stdev(n)
    """    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = (gray - np.mean(gray)) / (np.std(gray) + 1e-8) # normalization
    gray = np.dstack([gray]).astype(np.float32)
    return gray
        
