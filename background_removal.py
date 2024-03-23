# import cv2 to capture videofeed
import cv2

import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)
# setting framewidth and frameheight as 640 X 480
camera.set(3 , 640)
camera.set(4 , 480)

# loading the mountain image
mountain = cv2.imread('Mount-Everest.webp')

# resizing the mountain image as 640 X 480
mountain2 = cv2.resize(mountain,(640,480))

while True:

    # read a frame from the attached camera
    status , frame = camera.read()

    # if we got the frame successfully
    if status:

        # flip it
        frame = cv2.flip(frame , 1)

        # converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

        # creating thresholds
        lower_bound = np.array([100,100,100])
        upper_bound = np.array([255,255,255])
        
        # thresholding image
        mask = cv2.inRange(frame_rgb, lower_bound,upper_bound)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8)) 
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE,np.ones((3,3), np.uint8))

        # inverting the mask
        inverted_mask = cv2.bitwise_not(mask)
        # bitwise and operation to extract foreground / person
        foreground = cv2.bitwise_and(frame, frame, mask=inverted_mask)
        # final image
        new_frame = np.where(foreground==0,mountain2,foreground)
        
        cv2.imshow("cam",foreground)
        cv2.imshow('frame' , new_frame)

        code = cv2.waitKey(1)
        if code  ==  32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
