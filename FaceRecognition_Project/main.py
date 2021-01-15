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
from threading import Thread
#Fichier .py importé nécessaire au fonctionnement du programme

from Face_dataset import *
from get_face_from_image import *
from detection import *
from recognition import *
from tracking import *
from datetime import datetime

class CountsPerSec:
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        self._start_time = None
        self._num_occurrences = 0

    def start(self):
        self._start_time = datetime.now()
        return self

    def increment(self):
        self._num_occurrences += 1

    def countsPerSec(self):
        elapsed_time = (datetime.now() - self._start_time).total_seconds()
        return self._num_occurrences / elapsed_time

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True

class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True

def putIterationsPerSec(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """

    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame

def threadVideoGet(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    """

    video_getter = VideoGet(source).start()
    cps = CountsPerSec().start()

    while True:
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break

        frame = video_getter.frame
        frame = putIterationsPerSec(frame, cps.countsPerSec())
        cv2.imshow("Video", frame)
        cps.increment()

def threadVideoShow(source=0):
    """
    Dedicated thread for showing video frames with VideoShow object.
    Main thread grabs video frames.
    """

    cap = cv2.VideoCapture(source)
    (grabbed, frame) = cap.read()
    video_shower = VideoShow(frame).start()
    cps = CountsPerSec().start()

    while True:
        (grabbed, frame) = cap.read()
        if not grabbed or video_shower.stopped:
            video_shower.stop()
            break

        frame = putIterationsPerSec(frame, cps.countsPerSec())
        video_shower.frame = frame
        cps.increment()

def threadBoth(source="rtsp://192.168.1.20:8080/h264_ulaw.sdp"):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Dedicated thread for showing video frames with VideoShow object.
    Main thread serves only to pass frames between VideoGet and
    VideoShow objects/threads.
    """

    video_getter = VideoGet(source).start()
    video_shower = VideoShow(video_getter.frame).start()
    cps = CountsPerSec().start()

    while True:
        if video_getter.stopped or video_shower.stopped:
            video_shower.stop()
            video_getter.stop()
            break

        frame = video_getter.frame
        #frame = putIterationsPerSec(frame, cps.countsPerSec())
        video_shower.frame = frame
        cps.increment()

def main():
    choix = input(
        '\n 1) Face detection \n '
        '2) Face dataset + Face trainer \n '
        '3) Face recognition \n '    
        '4) Face recognition without tracking \n '
        '5) Zone tracking \n '
        '6) Skin detection \n '
        '7) Train face from folder \n'
        '8) Threading video \n'
        '9) Delete user \n')
    choix = int(choix)
    if choix == 1:
        face_detection() #detection de visage brut
        main()
    elif choix == 2:
        face_dataset()  #permet d'enregistrer 30 images d'un visage dans un dossier à son nom afin de pouvoir reconnaître son visage

        main()
    elif choix == 3:
        face_recognition() #reconnaissance du visage puis tracking du visage (limité à 1 visage)
        main()
    elif choix == 4:
        face_recognition_no_tracking() #reconnaissance du visage brut sans tracking (pas de limite de visage dans le flux vidéo)
        main()
    elif choix == 5:
        tracking_zone() #tracker une zone selectionné par l'utilisateur
        main()
    elif choix == 6:
        skin_detection() #detection de peau en passant dans l'espace HSV
        main()
    elif choix ==7:
       train_new_face()
       main()
    elif choix ==8: #flux vidéo en utilisant des thread
        main()
    elif choix ==9: #suppression d'un utilisateur
        delete_user()
        main()

    print('fin du programme')


main()
