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


def count_number_line():
    fichier = open("label.txt")
    nombre_ligne = len(fichier.readlines())
    return nombre_ligne


def face_dataset():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widths
    cam.set(4, 480)  # set video height

    face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

    # Entrée d'un id par personne
    face_id = count_number_line() + 1

    face_name = input('\n enter user name end press <return> ==>  ')
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")

    fichier = open("label.txt", "a")
    fichier.write(face_name + "\n")
    # variable permettant de savoir à quel itération nous sommes (30 fois le même visage sera capturé)
    count = 0

    os.mkdir("dataset/" + str(face_id) + str(face_name)) #créer un dossier au nom de la personne
    while (True):
     ret, img = cam.read()
     cv2.putText(img, " Nombre de photos prises : " +  str(count) + " /30", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
     cv2.putText(img, "Appuyez sur la barre espace pour prendre une photo", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
     cv2.imshow('image', img)
     k = cv2.waitKey(100) & 0xff  # Appuyez sur Echap pour quitter
     if k == 32:


        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            # Sauvegarde les visages dans le dossier correspondant au format .jpg
            cv2.imwrite("dataset/" + str(face_id) + str(face_name) + "/User." + str(face_id) + '.' + str(count) + ".jpg",gray[y:y + h, x:x + w])

        if count >= 30:  # Prend 30 screenshots en niveau de gris du visage détecté
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
    face_training("dataset")

def face_training(path):
    # chemin d'accès vers les visages


    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml");

    # Fonction permettant d'obtenir les images et les id associés en lisant le fichier contenu dans le dossier "path"
    def getImagesAndLabels(path):

        imagePaths=[]
        for i in os.listdir(path):

           imagePaths += [os.path.join(path + "/" + i , f) for f in os.listdir(path + "/" + i +"/")]

        faceSamples = []
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L')  # Conversion de l'image en niveau de gris
            img_numpy = np.array(PIL_img, 'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])


            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)

        return faceSamples, ids

    print("\n [INFO] Training : création/mise à jour du fichier trainer.yml")
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Sauvegarde du fichier trainer.yml dans le dossier correspondant
    recognizer.write('trainer/trainer.yml')

    # Print le nombre de visage enregistrés
    print("\n [INFO] {0} visages enregistrés. Fin du programme".format(len(np.unique(ids))))

