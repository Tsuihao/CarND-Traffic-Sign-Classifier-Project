import numpy as np

# Y' = 0.299 R + 0.587 G + 0.114 B 
# TODO: Move to preprocess folder
def rgb_to_gray(X_rgb):
    X_gray = []
    for sample in X_rgb:
        grayscale = np.dot(sample[...,:3], [0.299, 0.587, 0.114])
        X_gray.append(grayscale)
    
    # Turn into numpy ararry 
    X_gray = np.asarray(X_gray)
    X_gray = np.expand_dims(X_gray, axis=3)
    return X_gray