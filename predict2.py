import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout
from skimage.measure import compare_ssim as ssim

np.random.seed(0)
model = keras.models.load_model(r'C:\Users\nari9\OneDrive\Documents\PG\Assignments\Fall4\ComputerVision\Project\Data\model.hdf5')

t_data = np.load(r'C:\Users\nari9\OneDrive\Documents\PG\Assignments\Fall4\ComputerVision\Project\Data\data_x.npy', 'r')
o_data = np.load(r'C:\Users\nari9\OneDrive\Documents\PG\Assignments\Fall4\ComputerVision\Project\Data\data_y.npy', 'r')

t_data = t_data/255


outp = model.predict(t_data)
print (outp.shape)
#print (outp)

ss = 0

for i in range(len(outp)):
    ss += ssim(outp[i], o_data[i], multichannel = True)

print (ss/ len(outp))