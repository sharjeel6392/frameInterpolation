#In[0]:
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout, UpSampling2D
from sklearn.model_selection import train_test_split

t_data = np.load(r'/home/kirksalvator/Documents/Semesters/2020/Fall2020/CSCI 631 CV/Project/Data/data_x.npy', 'r')
o_data = np.load(r'/home/kirksalvator/Documents/Semesters/2020/Fall2020/CSCI 631 CV/Project/Data/data_y.npy', 'r')

print("Loading complete")
print (o_data.shape)

t_data = t_data/255.0
o_data = o_data/255.0


x_train = t_data
y_train = o_data

# In[2]:
model = Sequential()


model.add(Conv2D(32, kernel_size = (3,3) , input_shape = (128, 128, 6), activation = 'relu', padding = "same"))
model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = "same"))
model.add(MaxPool2D(pool_size = (2,2), padding = "same"))

model.add(Conv2D(64, kernel_size = (3,3) , input_shape = (128, 128, 6), activation = 'relu', padding = "same"))
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = "same"))
model.add(MaxPool2D(pool_size = (2,2), padding = "same"))


model.add(UpSampling2D(size = (2,2)))
model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = "same"))
model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = "same"))

model.add(UpSampling2D(size = (2,2)))
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = "same"))
model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = "same"))


model.add(Conv2D(3, kernel_size = (1,1), activation = 'relu', padding = "same"))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
print (model.summary())

# In[3]:
model.fit(x_train, y_train, epochs = 30)






# %%
model.save(r'/home/kirksalvator/Documents/Semesters/2020/Fall2020/CSCI 631 CV/Project/Data/model.hdf5')

# %%
