#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import streamlit as st
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[2]:


def detect_faces():

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Add instructions to the interface
    st.write('Press \'q\' to exit')
    st.write('Adjust the parameters to improve the detection')

    # Add a feature to allow the user to choose the color of the rectangles
    color = st.color_picker('Pick a color for the rectangles', '#00FF00')

    # Add a feature to adjust the minNeighbors parameter
    min_neighbors = st.slider('minNeighbors', 1, 10, 5)

    # Add a feature to adjust the scaleFactor parameter
    scale_factor = st.slider('scaleFactor', 1.1, 1.9, 1.3, 0.1)

    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()

        # Convert the frames to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the faces using the face cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Display the frames
        cv2.imshow('Face Detection using Viola-Jones Algorithm', frame)

        # Add a feature to save the images with detected faces on the user's device
        if st.button('Save Image'):
            filename = st.text_input('Enter file name', 'face_detection')
            cv2.imwrite(f'{filename}.jpg', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


# In[3]:


if __name__ == '__main__':
    detect_faces()


# In[ ]:




