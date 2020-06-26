import _pickle as cPickle
import cv2
import numpy as np
from os import walk
from wand.image import Image
import pytesseract
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import re

def blockparasing(input):    
    try:
        Entity = pytesseract.image_to_string(input)
    except Exception as e:
        Entity = []
        print(e)
    return Entity