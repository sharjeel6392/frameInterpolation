import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from skimage.filters import unsharp_mask


np.random.seed(0)
model = keras.models.load_model(r'/home/kirksalvator/Documents/Semester/2020/Fall2020/CSCI 631 CV/Project/Data/model.hdf5')


data = np.load(r'/home/kirksalvator/Documents/Semester/2020/Fall2020/CSCI 631 CV/Project/Data/indi_framedata.npy')
ctr = 0
data = data/255.0
print ('Normalised data')
#model = keras.models.load_model(r'C:\Users\nari9\OneDrive\Documents\PG\Assignments\Fall4\ComputerVision\Project\Data\model.hdf5')

for i in range(1, data.shape[0]):
  print (i)
  arr_in = np.ndarray((1, 128, 128, 6))
  arr_in[0,:,:,:3] = data[i-1]
  arr_in[0, :, :, 3:] = data[i]
  arr = model.predict(arr_in).reshape((128, 128, 3))
  #arr = unsharp_mask(arr, radius = 1, amount = 1)
  if ctr == 0:
    cv2.imwrite(r'/home/kirksalvator/Documents/Semester/2020/Fall2020/CSCI 631 CV/Project/OutputFrames/frame_' + str(ctr) + '.jpg', data[i-1]*255.0)
    cv2.imwrite(r'/home/kirksalvator/Documents/Semester/2020/Fall2020/CSCI 631 CV/Project/OutputFrames/frame_' + str(ctr+1) + '.jpg', arr*255.0)
    cv2.imwrite(r'/home/kirksalvator/Documents/Semester/2020/Fall2020/CSCI 631 CV/Project/OutputFrames/frame_' + str(ctr+2) + '.jpg', data[i]*255.0)
    ctr+=3
  else:
    cv2.imwrite(r'/home/kirksalvator/Documents/Semester/2020/Fall2020/CSCI 631 CV/Project/OutputFrames/frame_' + str(ctr) + '.jpg', arr*255.0)
    cv2.imwrite(r'/home/kirksalvator/Documents/Semester/2020/Fall2020/CSCI 631 CV/Project/OutputFrames/frame_' + str(ctr+1) + '.jpg', data[i]*255.0 )
    ctr+=2
