import cv2

cap = cv2.VideoCapture(r'/home/kirksalvator/Documents/Semester/2020/Fall2020/CSCI 631 CV/Project/input.mp4')

i = 0

while cap.isOpened():
    ret, frame = cap.read()

    if ret == False:
        break
    #grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(r'/home/kirksalvator/Documents/Semester/2020/Fall2020/CSCI 631 CV/Project/ExtractedFrames/frame_' + str(i) + '.jpg', frame)
    i+=1

cap.release()
cv2.destroyAllWindows()
