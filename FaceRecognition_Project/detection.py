import cv2
import numpy as np
from PIL import Image
import os
from random import randint
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time
import imutils
import sys

def face_detection(): #detection de visage brut
    faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            # redimensionnement de l'image  pour détecter des visages plus petits (approche pyramidale de viola & jones)
            minNeighbors=5,  # permet d'éliminer les faux positifs en appliquant une approche au voisinage
            minSize=(20, 20)  # visage minimum détectable (20 * 20 pixels)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

        cv2.imshow('video', img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:  # Appuyez sur ECHAP pour quitter
            break
    cap.release()
    cv2.destroyAllWindows()


def body_detection():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cv2.startWindowThread()

    cap = cv2.VideoCapture(0)

    while (True):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

        for (xA, yA, xB, yB) in boxes:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)



def skin_detection_image(frame):
    lower = np.array([0, 48, 80], dtype='uint8')
    upper = np.array([20, 255, 255], dtype='uint8')
    frame = imutils.resize(frame, width=400)
    size = np.size(frame)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)
    ret, thresh = cv2.threshold(skin, 0, 255, cv2.THRESH_BINARY)
    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    nonzeros = cv2.countNonZero(gray)
    print("white =")
    print(nonzeros)
    print("black =")
    print(size - nonzeros)
    cv2.imshow("images", np.hstack([frame, thresh]))
    if nonzeros > 40000:
        return True
    else:
        return False

#detection de peau brut
def skin_detection():
    lower = np.array([0, 48, 80], dtype='uint8')
    upper = np.array([20, 255, 255], dtype='uint8')
    camera = cv2.VideoCapture(0)

    while True:
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=400)
        size = np.size(frame) # nombre de pixel de l'image
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin = cv2.bitwise_and(frame, frame, mask=skinMask)
        ret, thresh = cv2.threshold(skin, 0, 255, cv2.THRESH_BINARY)
        gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        #cv2.imshow("images", skin)
        nonzeros = cv2.countNonZero(gray)
        print("white =")
        print(nonzeros) #nombre de pixel blanc
        print("black =")
        print(size - nonzeros) #nombre de pixel noir

        cv2.imshow("images", np.hstack([frame,thresh]))

        k = cv2.waitKey(10) & 0xff  # Appuyez sur Echap
        if k == 27:
            break
    camera.release()
    cv2.destroyAllWindows()


