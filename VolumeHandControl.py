import cv2
import time
import numpy as np
import HandTrackingModule as ht
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Setting camera dimensions
wCam, hCam = 640, 480

# Initializing the camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

# Initializing the hand detector
detector = ht.handDetector(detectionCon=0.7)

# Accessing the speaker device and volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Getting the volume range
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    if not success:
        break

    # Detecting hands
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]  # Extracting x and y values of the thumb
        x2, y2 = lmList[8][1], lmList[8][2]  # Extracting x and y values of the index finger
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Drawing circles and line between thumb and index finger
        cv2.circle(img, (x1, y1), 8, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 8, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

        # Calculating the length between the thumb and index finger
        length = math.hypot(x2 - x1, y2 - y1)
        print(length)

        # Interpolating the length to the volume range
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        volPer = np.interp(length, [50, 300], [0, 100])
        print(int(length), vol)

        # Setting the volume
        volume.SetMasterVolumeLevel(vol, None)

        if length < 50:
            cv2.circle(img, (cx, cy), 8, (0, 0, 255), cv2.FILLED)

    # Drawing the volume bar
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255,0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, ( 255,0, 0), 2)

    # Calculating the frame per second (FPS)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Displaying the FPS on the image
    cv2.putText(img, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    # Showing the image
    cv2.imshow("Image", img)

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
