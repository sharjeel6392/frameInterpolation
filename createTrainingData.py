import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

directory = r'/home/kirksalvator/Documents/Semesters/2020/Fall2020/CSCI 631 CV/Project/ExtractedFrames'
out = r'/home/kirksalvator/Documents/Semesters/2020/Fall2020/CSCI 631 CV/Project/Data/Sample'
data = np.ndarray((300, 128, 128, 3))
i = 0
print(directory + '/frame_0.jpg')
print (cv2.imread(directory + '/frame_0.jpg').shape)


for i in range(0,300):
    image =  cv2.imread(directory + '/frame_' + str(i) + '.jpg')
    res_image = cv2.resize(image, (128, 128))

    data[i] = res_image
    i+=1
    if i%50 ==  0:
        print (i)
        cv2.imwrite(r'/home/kirksalvator/Documents/Semester/2020/Fall2020/CSCI 631 CV/Project/Data/Sample/frame_' + str(i) + '.jpg', image)
        cv2.imwrite(r'/home/kirksalvator/Documents/Semester/2020/Fall2020/CSCI 631 CV/Project/Data/Sample/frame_resized_' + str(i) + '.jpg', res_image)

print (data.shape)
np.save(out, data)
