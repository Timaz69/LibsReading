from git import Repo
import numpy as np
import cv2
import os
from os.path import realpath, normpath

#Repo.clone_from('https://github.com/the-javapocalypse/Face-Detection-Recognition-Using-OpenCV-in-Python.git', r'C:\Users\Timaz\Documents\ECAM\ETS\MTI830-Forage de texte\projet\git')


#mouth_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_smile.xml')
print('Bonjour',cv2.__file__)
#mouth_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_smile.xml')
h = normpath(realpath(cv2.__file__) + '/../data/')
print(h)
mouth_cascade = cv2.CascadeClassifier(h+'/haarcascade_mouth.xml')
if mouth_cascade.empty():
    raise IOError('Unable to load the mouth cascade classifier xml file')


def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text,minSize=None):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
    #ret2, mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #res = cv2.bitwise_and(gray_img,gray_img, mask=mask)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors,minSize=minSize, flags=cv2.CASCADE_SCALE_IMAGE)
    coords = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

def draw_mouth(img,mask, classifier, scaleFactor, minNeighbors, color, text,minSize=None):
    # Converting image to gray-scale
    gray_img = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
    #ret2, mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #res = cv2.bitwise_and(gray_img,gray_img, mask=mask)
    # detecting features in gray-scale image, returns coordinates, width and height of features
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors,minSize=minSize, flags=cv2.CASCADE_SCALE_IMAGE)
    coords = []
    # drawing rectangle around the feature and labeling it
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords


# Method to detect the features
def detect(img, faceCascade, eyeCascade, mouthCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.3, 5, color['blue'], "Face")
    # If feature is detected, the draw_boundary method will return the x,y coordinates and width and height of rectangle else the length of coords will be 0
    if len(coords)==4:
        # Updating region of interest by cropping image
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        # Passing roi, classifier, scaling factor, Minimum neighbours, color, label text
        coords_e = draw_boundary(roi_img, eyeCascade, 1.3, 5, color['red'], "Eye")
        eye=roi_img.copy()
        if len(coords_e) ==4:
            eye[coords_e[1]:coords_e[1]+coords_e[3], coords_e[0]:coords_e[0]+coords_e[2]] = [0,0,0]
        #cv2.imshow("img",eye)
        masked = cv2.bitwise_and(roi_img, eye)
        cv2.imshow('masked eye',masked)
        coords_m = draw_mouth(roi_img,masked, mouthCascade, 1.1, 30, color['white'], "Mouth",minSize=(40,40))
        if len(coords_m) == 4 and len(coords_e) == 4:
            inside_x, inside_y = None,None
            for point in range(coords_m[0],coords_m[0]+coords_m[2]):
                print(coords_m[0])
                print(coords_e[0])
                if point>=0 and point<=14:
                    inside_x = True
                    print("oui")
                else:
                    inside_x = False
            for point in range(coords_m[1],coords_m[1]+coords_m[3]):
                if point>=coords_e[1] and point<=coords_e[1]+coords_e[3]:
                    inside_y = True
                else:
                    inside_y = False

            inside = inside_x & inside_y
            if inside:
                print("inside")

    return img


# Loading classifiers
faceCascade = cv2.CascadeClassifier(h+'/haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier(h+'/haarcascade_eye.xml')
mouthCascade = cv2.CascadeClassifier(h+'/haarcascade_mouth.xml')

# Capturing real time video stream. 0 for built-in web-cams, 0 or -1 for external web-cams
video_capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    # Reading image from video stream
    _, img = video_capture.read()
    # Call method we defined above
    img = detect(img, faceCascade, eyesCascade, mouthCascade)
    # Writing processed image in a new window
    cv2.imshow("face detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing web-cam
video_capture.release()
# Destroying output window
cv2.destroyAllWindows()
"""
"""