import pytesseract
import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import re
from src.creatingfolder import folder, gettingfilenames

def finding_counters(img):    
    blocks = []
    kernalnew = np.ones((1, 20), np.uint8)
    img_new = cv2.erode(img, kernalnew, iterations=15)
    binary_map1 = (img_new < 255).astype(np.uint8)
    blocks.append(cv2.connectedComponentsWithStats(binary_map1, 4, cv2.CV_32S)[2][1:])
    cv2.imwrite("3layout.png", img_new)
    return blocks[0]

def blockwiseparasing(blocks):    
    levels = []
    levels1 = []
    for i in range(len(blocks)):
        if blocks[i][-1] > 2500:
            levels.append(blocks[i][0])
            levels1.append(blocks[i])
    return np.array(levels), levels1

def intendation_finding(level1):    
    level = sorted(level1)
    listnew = []
    listnew.append(level[0])
    for i in range(1, len(level)):
        if level[i] - level[i-1] > abs(100)  :
            listnew.append(level[i])
    newlist = labeling_intendation(level1, listnew)
    return newlist

def labeling_intendation(level1, sortedlevel):    
    label_indent = []
    for i in range(len(level1)):
        label_i = []
        for j in range(len(sortedlevel)):
            label_i.append(abs(sortedlevel[j] - level1[i]))
        label_indent.append((level1[i], (np.argmin(label_i))))
    return label_indent

def cleaning(img_b):    
    block2 = []
    ret, img_b = cv2.threshold(img_b, 1, 255, cv2.THRESH_BINARY)
    horizontal = [sum(c) for c in zip(*img_b)]
    horizontal1 = [pixel / len(img_b) for pixel in horizontal]
    index = []
    for h in range(len(horizontal1)):
        if horizontal1[h] < 150:
            index.append(h)
            
    indexf = []
    indexf.append(index[0] - 10)
    for hw in range(1, len(index)):
        if index[hw] - index[hw-1] > 10:
            indexf.append(index[hw])
    indexf.append(index[-1])
    return indexf

def tocparsing(inputimage):    
    TOC = []
    img = cv2.imread(inputimage, 0)
    ret,img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    blocks = finding_counters(img)
    level, blocks1 = blockwiseparasing(blocks)
    listnew = intendation_finding(level)
    
    try:
        entity = pytesseract.image_to_string(img)
    except:
        entity = []
        
    Newentity = entity.split('\n')
    Newentity = list(filter(None, Newentity))
    for t in range(len(blocks1)):
        img_b = img[blocks1[t][1]:blocks1[t][1] + blocks1[t][3], :]
        ret, img_b = cv2.threshold(img_b, 1, 255, cv2.THRESH_BINARY)

        try:
            img_n = img[blocks1[t][1]:blocks1[t][1] + blocks1[t][3], :]
            img_p = img[blocks1[t][1]:blocks1[t][1] + blocks1[t][3], :]
            TOC.append((listnew[t][1], Newentity[t]))
        except:
            print(t)
    return TOC

if __name__ == '__main__':
    files = gettingfilenames('TOC/')
    Toc = []
    for k in range(len(files)):
        toc = tocparsing(files[k])
        Toc.extend(toc)
    tocdf = pd.DataFrame(Toc)
    tocdf.to_csv('newdharun.csv')