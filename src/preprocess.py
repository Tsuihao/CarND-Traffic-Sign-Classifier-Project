import numpy as np

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


#TODO
#- Add data augmentation