
import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from time import time


camera_video = cv2.VideoCapture(0)
 
# Set width of the frames in the video stream.
camera_video.set(3, 1280)
 
# Set height of the frames in the video stream.
camera_video.set(4, 720)
 
# Initialize a variable to store the time of the previous frame.
time1 = 0
 
# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():
    
    # Read a frame.
    ok, frame = camera_video.read()
    
    # Check if frame is not read properly.
    if not ok:
        
        # Continue to the next iteration to read the next frame.
        continue
 
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    
    # Desature the background of the frame.
    output_frame,_ = modifyBackground(frame, threshold = 0.3, display = False, method='desatureBackground')
    
    # Set the time for this frame to the current time.
    time2 = time()
    
    # Check if the difference between the previous and this frame time &gt; 0 to avoid division by zero.
    if (time2 - time1) &gt; 0:
    
        # Calculate the number of frames per second.
        frames_per_second = 1.0 / (time2 - time1)
        
        # Write the calculated number of frames per second on the frame. 
        cv2.putText(output_frame, 'fps: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 255, 0), 3)
    
    # Update the previous frame time to this frame time.
    # As this frame will become previous frame in next iteration.
    time1 = time2
    
    # Display the frame with desatured background.
    cv2.imshow('Video', output_frame)
    
    # Wait until a key is pressed.
    # Retreive the ASCII code of the key pressed
    k = cv2.waitKey(1) &amp; 0xFF
    
    # Check if 'ESC' is pressed.
    if (k == 27):
        
        # Break the loop.
        break
 
# Release the VideoCapture Object.
camera_video.release()
 
# Close the windows.
cv2.destroyAllWindows()