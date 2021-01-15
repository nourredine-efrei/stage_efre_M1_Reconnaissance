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
from recognition import *

def tracking(id, a1, a2, a3, a4):
    # Set up tracker.

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'CSRT']

    choix_algo = input(
        '\n 1) BOOSTING \n2) MIL \n3) KFC \n4) CSRT \n')
    choix_algo = int(choix_algo)
    if choix_algo == 1:
        tracker_type = tracker_types[0]
        tracker = cv2.TrackerBoosting_create()
    elif choix_algo == 2:
        tracker_type = tracker_types[1]
        tracker = cv2.TrackerMIL_create()
    elif choix_algo == 3:
        tracker_type = tracker_types[2]
        tracker = cv2.TrackerKCF_create()
    elif choix_algo == 4:
        tracker_type = tracker_types[3]
        tracker = cv2.TrackerCSRT_create()

    # Read video
    video = cv2.VideoCapture(0)

    # Exit if video not opened.
    if not video.isOpened():
        print
        "Could not open video"
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print
        'Cannot read video file'
        sys.exit()

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box

    bbox = (a1, a2, a3, a4)
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success

            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                        2)
            face_recognition()

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        cv2.putText(frame, str(id), (int(bbox[0]) + 5, int(bbox[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            break


def tracking_zone():
    # Set up tracker.

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'CSRT', 'TLD']

    choix_algo = input(
        '\n 1) BOOSTING \n2) MIL \n3) KFC \n4) CSRT \n 5) TLD \n')
    choix_algo = int(choix_algo)
    if choix_algo == 1:
        tracker_type = tracker_types[0]
        tracker = cv2.TrackerBoosting_create()
    elif choix_algo == 2:
        tracker_type = tracker_types[1]
        tracker = cv2.TrackerMIL_create()
    elif choix_algo == 3:
        tracker_type = tracker_types[2]
        tracker = cv2.TrackerKCF_create()
    elif choix_algo == 4:
        tracker_type = tracker_types[3]
        tracker = cv2.TrackerCSRT_create()
    elif choix_algo == 5:
        tracker_type = tracker_types[4]
        tracker = cv2.TrackerTLD_create()
    # Read video
    video = cv2.VideoCapture(0)

    # Exit if video not opened.
    if not video.isOpened():
        print
        "Could not open video"
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print
        'Cannot read video file'
        sys.exit()

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)  # permet de selectionner directement sur le flux vidéo la zone à tracker

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success

            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                        2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            break
