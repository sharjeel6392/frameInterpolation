import os
import cv2

writer = cv2.VideoWriter(filename="orig_video.mp4", fourcc=cv2.VideoWriter_fourcc(*'mp4v'), fps=30, frameSize=(512, 512))

for i in range(0, 598):
    print (i)
    img = cv2.imread(r'C:\Users\nari9\OneDrive\Documents\PG\Assignments\Fall4\ComputerVision\Project\SuperResolvedFrames\\frame_' + str(i) +'.jpg')
    #img = cv2.resize(img, (360, 640))
    writer.write(img)
    

writer.release()
