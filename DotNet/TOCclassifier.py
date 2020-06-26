import _pickle as cPickle
import cv2
import numpy as np
from os import walk
from wand.image import Image

def gettingfilenames(path):    
    f = []
    for (dirpath, dirnames, filenames) in walk(path):
        f.extend(filenames)
    return f

def reading_image(img):    
    img_n = cv2.imread(img, 0)
    img_r = cv2.resize(img_n, (256, 256))
    return img_r

def two_d_array_to_1d_array_covertion(inputarray):    
    one_d_array = []
    one_d_array.append(inputarray.flatten())
    return one_d_array

def data_augmentation(inputimage):    
    img_x = reading_image(inputimage)

    horizontal = [sum(c) for c in zip(*img_x)]
    horizontal1 = [pixel / len(img_x) for pixel in horizontal]
    horizontal1a = np.array(horizontal1)
    X = horizontal1a / 255
    X = np.expand_dims(X, axis=0)
    return X

def prediction(inputimage, model, i):    
    X_t = data_augmentation(inputimage)
    pred = model.predict(X_t)   
    return pred

def loading_model1():    
    with open('RFclassifier', 'rb') as f:
        rf = cPickle.load(f)
    return rf