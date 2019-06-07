import numpy as np
import pandas as pd
newlist = []
import os
from os import walk
from random import shuffle
import random
import numpy as np
from sklearn import preprocessing
import pandas as pd

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout,concatenate,Input,Conv1D, MaxPooling1D,Flatten
import pandas as pd
from keras.models import Model
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split,KFold

# The GPU id to use
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="3"
#Files of CSV we get from preprocessing
def gettingfilenames(path):
	f= []
	for (dirpath,dirnames,filenames) in walk(path):
		f.extend(filenames)
	return f
def meanvalue(array):
    horizantal = [sum(c) for c in zip(*array)]
    horizantal1 = [float(pixel / len(array)) for pixel in horizantal]
    horizantal1 = np.array(horizantal1)
    return horizantal1

def model():
    input_img = Input(shape=(100, 100, 1))
    model1 = Conv2D(50, kernel_size=(3, 3), activation='tanh', dilation_rate=(2, 2), padding='valid')(input_img)
    model1 = MaxPooling2D(pool_size=(2, 2))(model1)
    model1 = Dropout(0.1)(model1)
    model1 = Conv2D(50, kernel_size=(3, 3), activation='tanh', dilation_rate=(2, 2), padding='valid')(model1)
    model1 = MaxPooling2D(pool_size=(2, 2))(model1)
    model1 = Dropout(0.1)(model1)
    model1 = Conv2D(50, kernel_size=(3, 3), activation='tanh', dilation_rate=(2, 2), padding='valid')(model1)
    model1 = MaxPooling2D(pool_size=(2, 2))(model1)
    model1 = Dropout(0.1)(model1)
    outmodelf = Flatten()(model1)
    model = Dense(units=50, activation='tanh', input_dim=50, kernel_initializer='uniform')(outmodelf)
    model = Dense(units=5, activation='softmax', kernel_initializer='uniform')(model)
    model = Model(input_img, model)
    return model
files2 = gettingfilenames("text")
files1 = gettingfilenames("img")
files3 = gettingfilenames("table")
files4 = gettingfilenames("math")
files5 = gettingfilenames("lined")
##Data Balancing
newlist1 = []
newlist2 = []
newlist3 = []
newlist4 = []
newlist5 = []
leng = []
for g in range(0,len(files1)):
    df = pd.read_csv("img/%s" % str(files1[g]))
    print("img/%s" % str(files1[g]))
    x1 = df.values
    newlist1.extend(x1)

for g1 in range(0,len(files2)):
    df1 = pd.read_csv("text/%s" % str(files2[g1]))
    print("text/%s" % str(files2[g1]))
    x2 = df1.values
    newlist2.extend(x2)

for g2 in range(0,len(files3)):
    df2 = pd.read_csv("table/%s" % str(files3[g2]))
    print("table/%s" % str(files3[g2]))
    x3 = df2.values
    newlist3.extend(x3)
for g3 in range(0,len(files4)):
    df3 = pd.read_csv("math/%s" % str(files4[g3]))
    print("math/%s" % str(files4[g3]))
    x4 = df3.values
    newlist4.extend(x4)
for g4 in range(0,len(files5)):
    df4 = pd.read_csv("lined/%s"%str(files5[g4]))
    print("lined/%s" % str(files5[g4]))
    x5 = df4.values
    newlist5.extend(x5)

leng.append((len(newlist1),len(newlist2),len(newlist3),len(newlist4),len(newlist5)))
leng = sorted(leng[0])
print(leng[0])
newlist1n = random.sample(newlist1,leng[0])
newlist2n = random.sample(newlist2,leng[0])
newlist3n = random.sample(newlist3,leng[0])
newlist4n = random.sample(newlist4,leng[0])
newlist5n = random.sample(newlist5,leng[0])
newlist = newlist1n+newlist2n+newlist3n+newlist4n+newlist5n



newlist = random.sample(newlist, len(newlist))
X = []
Y = []
# Reshaping into 100 * 100 Boxes
for t in range(len(newlist)):

	x2 = np.reshape(newlist[t][1:-1], (-1, 100))
	y = newlist[t][-1]
	X.append(x2)
	Y.append(y)


# print("X abnd y done")


acc = []

kf = KFold(n_splits = 5)
train_index_list = []
test_index_list = []


print ("preprocessing done")





results = []
X = np.array(X)
Y = np.array(Y)
lb = preprocessing.LabelBinarizer()
Y = lb.fit_transform(Y)
FPR = []
TPR = []
F1score = []
precision = []
recall = []
index_train_l = []
index_test_l = []
for train_index, test_index in kf.split(X):
    index_train_l.append(train_index)
    index_test_l.append(test_index)
labelpredict = []
labeltest = []
for q in range(len(index_test_l)):
    # for train_index,test_index in kf.split(X):
    labelpredict = []
    labeltest = []

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test = X[index_train_l[q]], X[index_test_l[q]]
    y_train, y_test = Y[index_train_l[q]], Y[index_test_l[q]]

    model = Model()
    SGD = optimizers.SGD(lr=0.1)
    model.compile(optimizer=SGD, loss='mean_squared_error', metrics=['accuracy'])

    X1_test = np.expand_dims(X_test, axis=3)
    X1_train = np.expand_dims(X_train, axis=3)

    fmodel.fit(X1_train, y_train, validation_split=0.15, batch_size=32, epochs=200)


    y_pred = fmodel.predict(X1_test)
    labelpredict.extend(y_pred)
    labeltest.extend(y_test)
    labelpredictdf = pd.DataFrame(labelpredict)
    labelpredictdf.to_csv("labelpredictxATCNN%s.csv" % q)
    labeltestdf = pd.DataFrame(labeltest)
    labeltestdf.to_csv("labeltestxATCNN%s.csv" % q)

fmodel.save('DoTNet.h5')




