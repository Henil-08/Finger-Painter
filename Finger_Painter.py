import cv2 as cv
import numpy as np
import HandTrackingModule as htm
import os


def headerImages(path):
    img_path = os.listdir(path)
    image_list = []
    
    for images in img_path:
        image_list.append(cv.imread(f"{path}/{images}"))

    return image_list


def paintCondition(lmlist):
    fingercheck = detect.fingerCheck(lmlist)
    
    if fingercheck[1] and not fingercheck[0] and not fingercheck[2] and not fingercheck[3] and not fingercheck[4]:
        draw = True     
    elif fingercheck[1] and fingercheck[2] and not fingercheck[3] and not fingercheck[4]:
        draw = False
    else:
        draw = None
    
    return draw
   

if __name__ == "__main__":
    # Getting the images in the list
    path = 'Paint'
    image_list = headerImages(path)
    selection = image_list[0]

    # Initializing the webcam
    cap = cv.VideoCapture(0)
    cap.set(3, 1024)
    cap.set(4, 768)

    # Creating the HandDetector Object
    detect = htm.HandDetector(maxHands=1, detectConfi=0.95)  
    
    # Initializing the Drawing Parameters
    xp, yp = 0, 0
    imgcanvas = np.zeros((768, 1024, 3), np.uint8)
    brushThickness = 15
    brushColor = (0, 0, 255)
    eraserThickness = 75

    while (cap.isOpened()):
        isSucess, frame = cap.read()

        if not isSucess:
            print("No Frames Detected!")
            break
        
        # Fliping the frame horrizontally
        frame = cv.flip(frame, 1)

        # Detecting the hamds and getting the landmark list
        frame = detect.findHands(frame)
        lmlist = detect.findPosition(frame)

        if len(lmlist) != 0:
            draw = paintCondition(lmlist)
            
            # Getting the positions of Index and Middle Finger
            x1, y1 = lmlist[8][1:]
            x2, y2 = lmlist[12][1:]

            # If in Selection Mode
            if draw == False:
                xp, yp = 0, 0

                # For changing the header images
                if y1 < 149:
                    if 51 < x1 < 63:
                        brushColor = (0, 0, 255)
                        selection = image_list[0]
                    elif 189 < x1 < 252:
                        brushColor = (0, 255, 0)
                        selection = image_list[1]
                    elif 332 < x1 < 395:
                        brushColor = (255, 0, 255)
                        selection = image_list[2]
                    elif 470 < x1 < 533:
                        brushColor = (255, 0, 0)
                        selection = image_list[3]
                    elif 610 < x1 < 673:
                        brushColor = (51, 51, 102)
                        selection = image_list[4]
                    elif 751 < x1 < 814:
                        brushColor = (0, 102, 255)
                        selection = image_list[5]
                    elif 892 < x1 < 980:
                        brushColor = (0, 0, 0)
                        selection = image_list[6]

                cv.rectangle(frame, (x1, y1-25), (x2, y2+25), brushColor, cv.FILLED)
                cv.putText(frame, f"Selection Mode ON", (700, 200), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=brushColor, thickness=3)
   
            # For Drawing Mode
            if draw == True:
                # For First time there would be a line between Index Finger and Origin
                if xp==0 and yp==0:
                    xp, yp = x1, y1
                
                if brushColor == (0, 0, 0):
                    cv.circle(frame, (x1, y1), 15, brushColor, cv.FILLED)
                    cv.line(frame, (xp, yp), (x1, y1), brushColor, eraserThickness)
                    cv.line(imgcanvas, (xp, yp), (x1, y1), brushColor, eraserThickness)

                cv.circle(frame, (x1, y1), 15, brushColor, cv.FILLED)
                # cv.line(frame, (xp, yp), (x1, y1), brushColor, brushThickness)
                cv.line(imgcanvas, (xp, yp), (x1, y1), brushColor, brushThickness)
                cv.putText(frame, f"Drawing Mode ON", (700, 200), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=brushColor, thickness=3)

                # Updating the previous Positions
                xp, yp = x1, y1

        # Merging the frame and the Canvas
        imggray = cv.cvtColor(imgcanvas, cv.COLOR_BGR2GRAY)
        _, imgthres = cv.threshold(imggray, 50, 255, cv.THRESH_BINARY_INV)
        imgthres = cv.cvtColor(imgthres, cv.COLOR_GRAY2BGR)
        frame = cv.bitwise_and(frame, imgthres)
        frame = cv.bitwise_or(frame, imgcanvas)

        # Showing the Frames
        frame[0:169, 0:1024] = selection
        detect.addFPS(frame)
        cv.imshow("Frame", frame)
        # cv.imshow("Canvas", imgcanvas)
        # cv.imshow("thres", imgthres)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()
    