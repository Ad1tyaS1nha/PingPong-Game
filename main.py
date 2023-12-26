import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

#importing all images
imgBackground = cv2.imread("Resources/Background.png")
imgGameOver = cv2.imread("Resources/gameOver.png")
imgBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread("Resources/bat1.png", cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("Resources/bat2.png", cv2.IMREAD_UNCHANGED)

#Hand detector
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)


ballpos = [100,100]
speedX = 15
speedY = 15
gameOver = False
score = [0, 0]


while True:
    _, img = cap.read()
    imgRaw = img.copy()
    img = cv2.flip(img, 1)

    #find hand and its landmarks
    hands, img = detector.findHands(img, draw= True, flipType = False)

    #overlaying bg image
    img = cv2.addWeighted(img, 0.05, imgBackground, 0.95, 0)

    if hands:
        for hand in hands:
            x,y,w,h = hand["bbox"]
            h1, w1, _ = imgBat1.shape
            y1 = y-h1//2
            y1 = np.clip(y1, 20, 415)

            if hand['type']=="Left":
                img = cvzone.overlayPNG(img, imgBat1, (59,y1))
                if 59 < ballpos[0] < 59+w1 and y1 < ballpos[1] < y1+h1:
                    speedX = -speedX
                    ballpos[0] += 30
                    score[0] += 1

            if hand['type']=="Right":
                img = cvzone.overlayPNG(img, imgBat2, (1195,y1))
                if 1195-50 < ballpos[0] < 1195-30 and y1 < ballpos[1] < y1+h1:
                    speedX = -speedX
                    ballpos[0] -= 30
                    score[1] += 1

    #Gameover
    if ballpos[0] < 40 or ballpos[0] > 1200:
        gameOver = True

    if gameOver:
        img = imgGameOver
        cv2.putText(img, str(score[0]+score[1]).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX, 2.5, (200, 0, 200), 5)

    else:

        #moving the ball
        if ballpos[1] >= 500 or ballpos[1] <= 10:
            speedY = -speedY

        ballpos[0] += speedX
        ballpos[1] += speedY


    #draw the ball
    img = cvzone.overlayPNG(img, imgBall, ballpos)

    cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
    cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)

    img[580:700, 20:233] = cv2.resize(imgRaw, (213,120))

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        ballpos = [100, 100]
        speedX = 12
        speedY = 12
        gameOver = False
        score = [0, 0]
        imgGameOver = cv2.imread("Resources/gameOver.png")
