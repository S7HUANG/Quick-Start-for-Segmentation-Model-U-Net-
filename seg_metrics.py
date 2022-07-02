# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os

""" 
Confusion Matrix
P\L     P    N 
P      TP    FP 
N      FN    TN 
"""  
#  Get color dictionary
#  labelFolder (The reason for traversing folders is that a label may not contain all category colors)
#  classNum includes background
def color_dict(labelFolder, classNum):
    colorDict = []
    #  Get the name of the file in the folder
    ImageNameList = os.listdir(labelFolder)
    for i in range(len(ImageNameList)):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        img = cv2.imread(ImagePath).astype(np.uint32)
        # If grayscale, convert to RGB
        if(len(img.shape) == 2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.uint32)
        # To extract unique values, convert RGB to a number
        img_new = img[:,:,0] * 1000000 + img[:,:,1] * 1000 + img[:,:,2]
        unique = np.unique(img_new)
        # Add the unique value of the ith pixel matrix to the colorDict
        for j in range(unique.shape[0]):
            colorDict.append(unique[j])
        # The unique value in the current i pixel matrix is then taken as the unique value
        colorDict = sorted(set(colorDict))
        # If the number of unique values is equal to the total number of classes (including background) ClassNum, stop traversing the remaining images
        if(len(colorDict) == classNum):
            break
    # Store the BGR dictionary of colors for rendering results during prediction
    colorDict_BGR = []
    for k in range(len(colorDict)):
        #  Zeroes are added to the left for results that do not reach nine digits(eg:5,201,111->005,201,111)
        color = str(colorDict[k]).rjust(9, '0')
        # First 3 digits B, middle 3 digits G, last 3 digits R
        color_BGR = [int(color[0 : 3]), int(color[3 : 6]), int(color[6 : 9])]
        colorDict_BGR.append(color_BGR)
    # Convert to numpy format
    colorDict_BGR = np.array(colorDict_BGR)
    # Store the GRAY dictionary of colors for onehot encoding during preprocessing
    colorDict_GRAY = colorDict_BGR.reshape((colorDict_BGR.shape[0], 1 ,colorDict_BGR.shape[1])).astype(np.uint8)
    colorDict_GRAY = cv2.cvtColor(colorDict_GRAY, cv2.COLOR_BGR2GRAY)
    return colorDict_BGR, colorDict_GRAY

def ConfusionMatrix(numClass, imgPredict, Label):  
    # Return Confusion Matrix
    mask = (Label >= 0) & (Label < numClass)  
    label = numClass * Label[mask] + imgPredict[mask]  
    count = np.bincount(label, minlength = numClass**2)  
    confusionMatrix = count.reshape(numClass, numClass)  
    return confusionMatrix

def OverallAccuracy(confusionMatrix):  
    # Returns the overall pixel precision OA of all classes
    # acc = (TP + TN) / (TP + TN + FP + TN)  
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()  
    return OA
  
def Precision(confusionMatrix):  
    # Returns the precision rate for all categories
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    return precision  

def Recall(confusionMatrix):
    # Returns the recall rate for all categories
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    return recall
  
def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 1)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis = 0)
    f1score = 2 * precision * recall / (precision + recall)
    return f1score

def IntersectionOverUnion(confusionMatrix):  
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    return IoU

def MeanIntersectionOverUnion(confusionMatrix):  
    intersection = np.diag(confusionMatrix)  
    union = np.sum(confusionMatrix, axis = 1) + np.sum(confusionMatrix, axis = 0) - np.diag(confusionMatrix)  
    IoU = intersection / union
    mIoU = np.nanmean(IoU)  
    return mIoU
  
def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)  
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis = 1) +
            np.sum(confusionMatrix, axis = 0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

#################################################################
LabelPath = r"Data\test\label1"
PredictPath = r"Data\test\predict1"
classNum = 13
#################################################################

# Get category color dictionary
colorDict_BGR, colorDict_GRAY = color_dict(LabelPath, classNum)

# Get all images in a folder
labelList = os.listdir(LabelPath)
PredictList = os.listdir(PredictPath)

# Retrieve the first image, the shape of which will be used later
Label0 = cv2.imread(LabelPath + "//" + labelList[0], 0)

# Number of images
label_num = len(labelList)

# Put all images in an array
label_all = np.zeros((label_num, ) + Label0.shape, np.uint8)
predict_all = np.zeros((label_num, ) + Label0.shape, np.uint8)
for i in range(label_num):
    Label = cv2.imread(LabelPath + "//" + labelList[i])
    Label = cv2.cvtColor(Label, cv2.COLOR_BGR2GRAY)
    label_all[i] = Label
    Predict = cv2.imread(PredictPath + "//" + PredictList[i])
    Predict = cv2.cvtColor(Predict, cv2.COLOR_BGR2GRAY)
    predict_all[i] = Predict

# Mapping colors to 0,1,2,3...
for i in range(colorDict_GRAY.shape[0]):
    label_all[label_all == colorDict_GRAY[i][0]] = i
    predict_all[predict_all == colorDict_GRAY[i][0]] = i

# Straighten into one dimension
label_all = label_all.flatten()
predict_all = predict_all.flatten()

# Calculate the confusion matrix and each accuracy parameter
confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
precision = Precision(confusionMatrix)
recall = Recall(confusionMatrix)
OA = OverallAccuracy(confusionMatrix)
IoU = IntersectionOverUnion(confusionMatrix)
FWIOU = Frequency_Weighted_Intersection_over_Union(confusionMatrix)
mIOU = MeanIntersectionOverUnion(confusionMatrix)
f1ccore = F1Score(confusionMatrix)

for i in range(colorDict_BGR.shape[0]):
    # Output category colors, need to install webcolors, direct pip install webcolors
    try:
        import webcolors
        rgb = colorDict_BGR[i]
        rgb[0], rgb[2] = rgb[2], rgb[0]
        print(webcolors.rgb_to_name(rgb), end = "  ")
    # Output grayscale value if not installed
    except:
        print(colorDict_GRAY[i][0], end = "  ")
print("")
print("Confusion Matrix")
print(confusionMatrix)
print("Precision:")
print(precision)
print("Recall")
print(recall)
print("F1-Score:")
print(f1ccore)
print("Overall Accuracy")
print(OA)
print("IoU:")
print(IoU)
print("mIoU:")
print(mIOU)
print("FWIoU:")
print(FWIOU)