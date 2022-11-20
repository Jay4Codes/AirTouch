from time import sleep
import cv2
import handTrackingModule as htm
import numpy as np
import autopy
import pyautogui
import math
import time

cap=cv2.VideoCapture(0)
wCam,hCam=640,480
cap.set(3,wCam)
cap.set(4,hCam)
detector = htm.handdetector(maxHands=1)
wScr, hScr = autopy.screen.size() 
#1536.0 864.0
frameR=100
smooth=7

plocX, plocY = 0, 0
clocX,clocY = 0, 0 
prev_y=0
prev_x=0

while True:
    success, img =cap.read()
    img= detector.findHands (img)
    LmList, bbox = detector.Position(img)
    cv2.rectangle (img, (frameR, frameR), (wCam-frameR, hCam-frameR),(255, 0, 255), 2)
    #cv2.rectangle (img, (frameR+50, frameR), (wCam-frameR-50, hCam-frameR-50),(255, 0, 255), 2)
                                                
    if len(LmList)!=0:
      x1, y1 = LmList[8][1:]
      x2, y2 =LmList[12][1:] 
      
      fingers=detector.fingersup()
      print(fingers)


# move
      if fingers[1]==1 and fingers[2] ==0 and fingers[0]==0:
       x3 =np.interp(x1, (frameR+50, wCam-frameR-50), (0, wScr))
       y3=np.interp(y1, (frameR, hCam-frameR-50), (0, hScr))
      #  x3 =np.interp(x1, (frameR+50, wCam-frameR-50), (0, wScr))
      #  y3=np.interp(y1, (frameR, hCam-frameR-50), (0, hScr))
       clocX =plocX +(x3 -plocX) /smooth
       clocY =plocY+(y3 -plocY) /smooth
       
       autopy.mouse.move(wScr-clocX,clocY)
       plocX, plocY =clocX, clocY 

       cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

      if fingers[1]==1 and fingers[2] ==1 and fingers[0]==0 and fingers[3]==0 and fingers[4]==0:
          Length, img,lineinfo=detector.findDistance(8, 12, img) 
          #print(Length)

          if Length <25:
            cv2.circle(img, (lineinfo[4],lineinfo[5]), 15, (0, 255, ), cv2.FILLED)
            pyautogui.click(clicks=2)
            time.sleep(0.5)

      if fingers[1]==1 and fingers[0] ==0 and fingers[4]==1 and fingers[3]==0 and fingers[2]==0:
            pyautogui.click(clicks=1)
            time.sleep(0.5)
   #scroller
      if fingers[0]==0 and fingers[1]==1 and fingers[2]==1 and fingers[3]==1 and fingers[4]==1:
        x1, y1 = LmList[12][1], LmList[12][2]
        if y1 -prev_y > 1:#5
                pyautogui.scroll(150)
        elif prev_y - y1 > 1:
            pyautogui.scroll(-150)
        prev_y = y1


      if fingers[4]==1 and fingers[:4]==[0,0,0,0]:
             pyautogui.click(button='right')
             time.sleep(0.5)


     #Drag
      
      # if fingers[0]==1 and fingers[1]==1 : 
      #   length,img,_=detector.findDistance(4,8,img)
      #   # print(length)  
      #   if length<20:  
      #     x3 =np.interp(x1, (frameR, wCam-frameR), (0, wScr))
      #     y3=np.interp(y1, (frameR, hCam-frameR), (0, hScr))

      #     clocX =plocX +(x3 -plocX) /smooth
      #     clocY =plocY+(y3 -plocY) /smooth
                
      #     pyautogui.drag(x1,y1, button='left') 
      #     plocX, plocY =clocX, clocY  
 
  # scrollHorizontal
      # if fingers[1]==1 and fingers[0]==1 and fingers[2:]==[0,0,0]:
      #       x1, y1 = LmList[8][1], LmList[8][2]
      #   if x1 -prev_x > 5:
      #     pyautogui.keyDown('shift')
      #     pyautogui.keyDown('ctrl')
      #     pyautogui.scroll(120)
      #   elif prev_x- x1 > 5:
      #      pyautogui.keyDown('shift')
      #      pyautogui.keyDown('ctrl')
      #      pyautogui.scroll(-120)
      #   prev_x = x1
      # pyautogui.keyUp('shift') 
      # pyautogui.keyUp('ctrl')  
    cv2.imshow("img",img) 
    cv2.waitKey(1)
