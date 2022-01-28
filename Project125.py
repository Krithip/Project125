import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time
x = np.load('C:/Users/Krithi/Desktop/Python/image.npz')["arr_0"]
y = pd.read_csv('C:/Users/Krithi/Desktop/Python/labels.csv')["labels"]
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H','I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, random_state = 9, train_size = 3500, test_size = 500)
xtrainScaled = xtrain/255.0
xtestScaled = xtest/255.0
classifier = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(xtrainScaled, ytrain)
def getPrediction(image):
    im_pil = Image.open(image)
    image_bw = im_pil.convert('L')
    imageReSized = image_bw.resize((28, 28), Image.ANTIALIAS)
    pixalFilter = 20
    minPixal = np.percentile(imageReSized, pixalFilter)
    imageInverted = np.clip(imageReSized - minPixal, 0, 255)
    maxPixal = np.max(imageReSized)
    imageInverted = np.asarray(imageInverted)/maxPixal
    testSample = np.array(imageInverted).reshape(1, 784)
    testPredict = classifier.predict(testSample)
    return testPredict[0]