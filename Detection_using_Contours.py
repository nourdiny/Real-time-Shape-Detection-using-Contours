import cv2
import numpy as np




#########################
wImg = 440
hImg = 280
cap = cv2.VideoCapture(1)
cap.set(3,wImg)
cap.set(4,hImg)
###########################
def empty(a):
    pass
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("threshold1","Parameters",150,255,empty)
cv2.createTrackbar("threshold2","Parameters",150,255,empty)
cv2.createTrackbar("Area","Parameters",1000,9000,empty)
nou =  False
#nou = True
def getConteurs(img,imgConteur):
    conteurs , hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in conteurs :
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area","Parameters")
        if area > areaMin:
            cv2.drawContours(imgConteur,cnt,-1,(255,0,255),2)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri,True)
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(imgConteur,( x , y ),( x + w , y + h),
                          (0,255,0),2)
            cv2.putText(imgConteur,"Points: "+str(len(approx)),
                        (x , y+h + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        .7,(0,255,0),2)
            cv2.putText(imgConteur, "Area: " + str(area),
                        (x, y + h + 40), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        .7, (0, 255, 0), 2)



while True:
    if nou :
        susse, img = cap.read()
    else:
        img = cv2.imread("img/img.JPG")
    img = cv2.resize(img , (wImg,hImg))
    imgConteur = img.copy()
    imgBlur = cv2.GaussianBlur(img,(7,7),1)
    imgGray = cv2.cvtColor(imgBlur,cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("threshold1","Parameters")
    threshold2 = cv2.getTrackbarPos("threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray,threshold1,threshold2)
    kernel = np.ones((3,3))
    imgDil = cv2.dilate(imgCanny,kernel,iterations=1)
    getConteurs(imgDil,imgConteur)


    ###########################3
    imgGray = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR)
    imgCanny = cv2.cvtColor(imgCanny, cv2.COLOR_GRAY2BGR)
    imgDil = cv2.cvtColor(imgDil, cv2.COLOR_GRAY2BGR)
    imgV1 = np.hstack([img,imgBlur,imgGray])
    imgV2 = np.hstack([imgCanny,imgDil,imgConteur])
    #####################

    cv2.imshow("imgStac",imgV1)
    cv2.imshow("img", imgV2)
    if cv2.waitKey(1) & 0xFF == ord('n'):
        break
