
import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from time import time


mp_selfie_segmentation = mp.solutions.selfie_segmentation
 
# Setting up Segmentation function.
segment = mp_selfie_segmentation.SelfieSegmentation(0)

camera_video = cv2.VideoCapture(0)
 
# Set width of the frames in the video stream.
camera_video.set(3, 1280)
 
# Set height of the frames in the video stream.
camera_video.set(4, 720)
 
# Initialize a variable to store the time of the previous frame.
time1 = 0




def modifyBackground(image, background_image = 255, blur = 95, threshold = 0.3, display = True, method='changeBackground'):
    '''
    This function will replace, blur, desature or make the background transparent depending upon the passed arguments.
    Args:
        image: The input image with an object whose background is required to modify.
        background_image: The new background image for the object in the input image.
        threshold: A threshold value between 0 and 1 which will be used in creating a binary mask of the input image.
        display: A boolean value that is if true the function displays the original input image and the resultant image 
                 and returns nothing.
        method: The method name which is required to modify the background of the input image.
    Returns:
        output_image: The image of the object from the input image with a modified background.
        binary_mask_3: A binary mask of the input image. 
    '''
 
    # Convert the input image from BGR to RGB format.
    RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
    # Perform the segmentation.
    result = segment.process(RGB_img)
    
    # Get a binary mask having pixel value 1 for the object and 0 for the background.
    # Pixel values greater than the threshold value will become 1 and the remainings will become 0.
    binary_mask = result.segmentation_mask > threshold
    
    # Stack the same mask three times to make it a three channel image.
    binary_mask_3 = np.dstack((binary_mask,binary_mask,binary_mask))
    
    if method == 'changeBackground':
    
        # Resize the background image to become equal to the size of the input image.
        background_image = cv2.resize(background_image, (image.shape[1], image.shape[0]))
 
        # Create an output image with the pixel values from the original sample image at the indexes where the mask have 
        # value 1 and replace the other pixel values (where mask have zero) with the new background image.
        output_image = np.where(binary_mask_3, image, background_image)
        
    elif method == 'blurBackground':
        
        # Create a blurred copy of the input image.
        blurred_image = cv2.GaussianBlur(image, (blur, blur), 0)
 
        # Create an output image with the pixel values from the original sample image at the indexes where the mask have 
        # value 1 and replace the other pixel values (where mask have zero) with the new background image.
        output_image = np.where(binary_mask_3, image, blurred_image)
    
    elif method == 'desatureBackground':
        
        # Create a gray-scale copy of the input image.
        grayscale = cv2.cvtColor(src = image, code = cv2.COLOR_BGR2GRAY)
 
        # Stack the same grayscale image three times to make it a three channel image.
        grayscale_3 = np.dstack((grayscale,grayscale,grayscale))
 
        # Create an output image with the pixel values from the original sample image at the indexes where the mask have 
        # value 1 and replace the other pixel values (where mask have zero) with the new background image.
        output_image = np.where(binary_mask_3, image, grayscale_3)
        
    elif method == 'transparentBackground':
        
        # Stack the input image and the mask image to get a four channel image. 
        # Here the mask image will act as an alpha channel. 
        # Also multiply the mask with 255 to convert all the 1s into 255.  
        output_image = np.dstack((image, binary_mask * 255))
        
    else:
        # Display the error message.
        print('Invalid Method')
        
        # Return
        return
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
    
    # Otherwise
    else:
        
        # Return the output image and the binary mask.
        # Also convert all the 1s in the mask into 255 and the 0s will remain the same.
        # The mask is returned in case you want to troubleshoot.
        return output_image, (binary_mask_3 * 255).astype('uint8')
 
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
    output_frame,_ = modifyBackground(frame, threshold = 0.3, display = False, method='blurBackground')
    
    # Set the time for this frame to the current time.
    time2 = time()
    
    # Check if the difference between the previous and this frame time &gt; 0 to avoid division by zero.
    if (time2 - time1) > 0:
    
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
    k = cv2.waitKey(1) and 0xFF
    
    # Check if 'ESC' is pressed.
    if (k == 27):
        
        # Break the loop.
        break
 
# Release the VideoCapture Object.
camera_video.release()
 
# Close the windows.
cv2.destroyAllWindows()