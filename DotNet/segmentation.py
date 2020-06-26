import os
import cv2
import numpy as np
from wand.image import Image
import re
from src.BlockPrasing import blockparasing

def undesired_objects(image):    
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[0]
    for i in range(1, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.ones(output.shape)
    img2[output == max_label] = 255
    return img2

def two_d_array_to_1d_array_covertion(inputarray):    
    one_d_array = []
    one_d_array.append(inputarray.flatten())
    return one_d_array

def normalazation(twod_array):    
    new_twod_array = [[float(pixel / 255) for pixel in subarray] for subarray in twod_array]
    return new_twod_array

def counting_components(connectivity, inputfile):    
    src = cv2.imread(inputfile, 0)
    binary_map = (src < 255).astype(np.uint8)
    output = cv2.connectedComponentsWithStats(binary_map, connectivity, cv2.CV_32S)
    return output[0]

def horizontal_segmentation(img_new):
    kernalnew = np.ones((0, 20), np.uint8)
    img_newx = cv2.erode(img_new, kernalnew, iterations=2)
    return img_newx

def data_augmentation(img_x):    
    print(len(img_x), len(img_x[0]))
    
    img_list1 = []
    if len(img_x) > 300:
        if len(img_x[0]) > 300:
            x1 = int((len(img_x) - 100) / 30)
            y1 = int((len(img_x[0]) - 100) / 30)
            for x in range(0, x1):
                for y in range(0, y1):
                    newimg_list = []
                    img_list = []
                    img_list = img_x[30*(x):100 + 30*(x), 30*(y):100 + 30*(y)]

                    img_list_array = two_d_array_to_1d_array_covertion(img_list)

                    X = np.append(img_list_array[0], 0)

                    img_list1.append(X)
        else:
            img_x = np.concatenate((img_x, img_x), axis=1)
            img_x = np.concatenate((img_x, img_x), axis=1)
            x1 = int((len(img_x) - 200) / 30)
            y1 = int((len(img_x[0]) - 200) / 30)
            for x in range(0, x1):
                for y in range(0, y1):
                    img_list = []
                    newimg_list = []
                    img_list = img_x[30*(x):100 + 30*(x), 30*(y):100 + 30*(y)]

                    img_list_array = two_d_array_to_1d_array_covertion(img_list)
                    X = np.append(img_list_array[0], 0)

                    img_list1.append(X)


    else :
        if len(img_x[0]) > 200:
            img_x = np.concatenate((img_x, img_x), axis=0)
            img_x = np.concatenate((img_x, img_x), axis=0)
            img_x = np.concatenate((img_x, img_x), axis=0)
            img_x = np.concatenate((img_x, img_x), axis=0)
            
            x1 = int((len(img_x) - 100) / 30)
            y1 = int((len(img_x[0]) - 100) / 30)
            for x in range(0, x1):
                for y in range(0, y1):
                    newimg_list = []
                    img_list = []
                    img_list = img_x[30*(x):100 + 30*(x), 30*(y):100 + 30*(y)]

                    img_list_array = two_d_array_to_1d_array_covertion(img_list)

                    X = np.append(img_list_array[0], 0)

                    img_list1.append(X)
        else:

            img_x = np.concatenate((img_x, img_x), axis=0)
            img_x = np.concatenate((img_x, img_x), axis=1)
            img_x = np.concatenate((img_x, img_x), axis=0)
            print(len(img_x), len(img_x[0]))
            x1 = int((len(img_x) - 100) / 30)
            y1 = int((len(img_x[0]) - 100) / 30)
            for x in range(0, x1):
                for y in range(0, y1):
                    img_list = []
                    newimg_list = []
                    img_list = img_x[30*(x):100 + 30*(x), 30*(y):100 + 30*(y)]

                    img_list_array = two_d_array_to_1d_array_covertion(img_list)


                    X = np.append(img_list_array[0], 0)

                    img_list1.append(X)
    img_list1 = [[float(pixel) / 255 for pixel in subarray] for subarray in img_list1]
    return img_list1

def ourapproach(model, connectivity, inputfile, num, name):    
    blocks = []
    img_i = cv2.imread(inputfile, 0)
    img_new = cv2.imread(inputfile, 0)
    
    list1 = []
    kernalnew = np.ones((2, 4), np.uint8) 
    kernalnew1 = np.ones((1, 1), np.uint8)
    output = counting_components(connectivity, inputfile)
    img_new = cv2.erode(img_new, kernalnew, iterations=25)
    for yf in range(0, 1000):

        img_new = cv2.erode(img_new, kernalnew, iterations=1)
        img_new = cv2.dilate(img_new, kernalnew1, iterations=1)
        binary_map1 = (img_new < 255).astype(np.uint8)
        output1 = cv2.connectedComponentsWithStats(binary_map1, connectivity, cv2.CV_32S)[0]

        if output > output1:
            output = output1

        else:

            img_new = cv2.dilate(img_new, kernalnew1, iterations=2)
            cv2.imwrite("%s_layout.png" % inputfile, img_new)
            binary_map1 = (img_new < 200).astype(np.uint8)
            blocks.append(cv2.connectedComponentsWithStats(binary_map1, connectivity, cv2.CV_32S)[2][1:])
            Ent = []
            for t in range(1, len(blocks[0])):
                img_b = img_i[blocks[0][t][1]:blocks[0][t][1] + blocks[0][t][3], blocks[0][t][0]:blocks[0][t][0] + blocks[0][t][2]]
                for ii in range(len(img_b)):
                    for jj in range(len(img_b[0])):
                        if img_b[ii][jj] > 215:
                            img_b[ii][jj] = 215
                        else:
                            img_b[ii][jj] = img_b[ii][jj]
                            
                img_x = img_b
                x1 = data_augmentation(img_x)
                
                try:

                    X = []
                    Y = []
                    pred = []
                    label_count = []
                    for m in range(len(x1)):
                        x2 = np.reshape(x1[m][0:-1], (-1, 100))
                        y = x1[m][-1]

                        X.append(x2)
                        Y.append(y)
                        
                    X = np.array(X)
                    X1 = np.expand_dims(X, axis=3)
                    y_pred = model.predict(X1)
                    
                    for r in range(len(y_pred)):
                        label = np.argmax(y_pred[r])

                        pred.append(label)
                        table = pred.count(2)
                        image = pred.count(0)
                        text = pred.count(1)
                        math = pred.count(3)
                        lined = pred.count(4)
                        label_count = [image, text, table, math, lined]
                        lab = label_count.index(max(label_count))
                        
                    print(lab)
                    if lab == 0:
                        print('image')
                    elif lab == 1:
                        print('text')
                        Entity = blockparasing(img_x)
                        print(Entity)
                        Ent.append(str(Entity))
                        Ent.append('\n')
                    elif lab == 2:
                        print('table')
                    elif lab == 3:
                        print('math')
                        Entity = blockparasing(img_x)
                        Ent.append(str(Entity))
                        Ent.append('\n')
                    else:
                        print('Lined')

                except:
                    print('noise')

            return Ent