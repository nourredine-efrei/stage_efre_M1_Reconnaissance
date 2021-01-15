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
from Face_dataset import *
def getImages(path,face_name, face_id):

    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    faceSamples = []
    ids = []
    detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml");
    count=0

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L')  # Conversion de l'image en niveau de gris
        img_numpy = np.array(PIL_img, 'uint8')


        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            cv2.imwrite("dataset/" + str(face_id) + str(face_name) +"/User." + str(face_id) + '.' + str(count) + ".jpg", img_numpy[y:y + h, x:x + w])
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(face_id)
            count+=1





    return faceSamples, ids

def train_new_face():
    nombre_ligne = count_number_line()



    face_id = count_number_line() + 1
    fichier = open("label.txt", "a")
    face_name = input('\n enter user name end press <return> ==>  ')
    fichier.write(face_name + "\n")
    os.mkdir("dataset/" + str(face_id) + str(face_name))  # créer un dossier au nom de la personne
    faces, ids = getImages('photo',face_name, face_id)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces,np.array(ids))
    recognizer.write('trainer/trainer.yml')


def delete_user():

    # Ouvrir le fichier en lecture seule
    print("Quel utilisateur souhaitez vous supprimer ?")
    file = open('label.txt', "r")
    taille_liste=1
    for line in file:
        print(str(taille_liste) + ") " +line)
        taille_liste+=1
    file.close()

    entree= input("entrez l'id associé au nom de la personne souhaitée : ")

    fichier=open('label.txt', "r")
    a = 0

    while str(entree)!= str(a):
        line = fichier.readline()
        a+=1
    fichier.close()

    print("Voulz vous vraiment supprimer " + line + " ?")
    print("1) oui")
    print("2) non")
    choix=input()
    if choix == str(1): #oui
        id_personne=0
        flag=0
        for i in os.listdir("dataset"):
            nom = line.rstrip('\n') #nom lu dans le fichier label sans le \n
            nom_id = str(entree) + str(nom) # nom associé à l'id correspondant au nom du dossier

            if int(entree) - 1 <= taille_liste - 1 and flag == 1:
                fichier = open("label.txt", "r")
                b = 0

                while b != id_personne:
                    nom_personne = fichier.readline().rstrip('\n')
                    b += 1
                nom_personne = fichier.readline().rstrip('\n')
                c=1
                for k in os.listdir("dataset/" + str(i) ):
                    os.rename(("dataset/" + str(i) + "/" + str(k)), "dataset/" + str(i) + "/" + "User." + str(id_personne) +"." + str(c) + ".jpg"  )
                    c+=1
                os.rename("dataset/" + i, "dataset/" + str(id_personne) + nom_personne)

            id_personne += 1

            if i == nom_id:
                for j in os.listdir("dataset/" + nom_id):
                    os.remove("dataset/" + nom_id + "/" + j)  #supprime les images dans le dossier
                os.rmdir("dataset/" +nom_id)   #supprime le dossier
                flag=1







        fichier2= open("label.txt", "r")
        lines= fichier2.readlines()
        fichier2.close()
        fichier2 = open("label.txt", "w")
        dernier_nom= len(lines)
        for line in lines:

            if str(line) != str(nom) + "\n" and str(line) != lines[dernier_nom -1]:
                fichier2.write(line)
        if lines[dernier_nom - 1] != str(nom):
            fichier2.write(line)

        fichier2.close()

        face_training("dataset")


    if choix== str(2):
        return 0




