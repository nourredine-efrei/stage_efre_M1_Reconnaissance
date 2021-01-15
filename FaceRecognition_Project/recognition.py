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
from detection import *
from tracking import tracking
def face_recognition_no_tracking():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "Cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialisation de l'id à 0
    id = 0

    # Initialisation de la capture vidéo
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    # définition de la taille minimal de la fenêtre où l'ont peut reconnaitre un visage (voir explication sur minSize de la fonction detectMultiScale)
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:

        ret, img = cam.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            img_crop = img[y:y + h, x:x + w]
            if not (skin_detection_image(img_crop)):
                break;

            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Plus la variable confidence est basse, plus le visage prédit ressemble à la personne détecté
            if (confidence < 100):
                fichier = open("label.txt", "r")
                a = 0
                while a != id:
                    line = fichier.readline()
                    a += 1

                id = line
                confidence_value = round(100 - confidence)
                confidence = "  {0}%".format(round(100 - confidence))
                if (confidence_value < 20): # plus cette valeur sera grande, plus le taux de tolérance est grand (mais il y aura donc plus d'erreur aussi)
                    id = "unknown"

            else:

                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  # Appuyez sur Echap
        if k == 27:
            break


def face_recognition():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "Cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialisation de l'id à 0
    id = 0

    # Initialisation de la capture vidéo
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    # définition de la taille minimal de la fenêtre où l'ont peut reconnaitre un visage (voir explication sur minSize de la fonction detectMultiScale)
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:

        ret, img = cam.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Plus la variable confidence est basse, plus le visage prédit ressemble à la personne détecté
            if (confidence < 100):
                fichier = open("label.txt", "r")
                a = 0
                while a != id:
                    line = fichier.readline()
                    a += 1

                id = line

                confidence = "  {0}%".format(round(100 - confidence))
                tracking(id, x, y, w,
                         h)  # lance la fonction tracking en trackant les coordonnées obtenue par la reconnaissance de visage
            else:
                img_crop = img[y:y + h, x:x + w]
                skin_detection_image(img_crop)
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  # Appuyez sur Echap
        if k == 27:
            break

    print("\n [INFO] Sortie du programme")
    cam.release()
    cv2.destroyAllWindows()