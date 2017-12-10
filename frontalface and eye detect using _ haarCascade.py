import cv2
import numpy as np
#haarcascade file paths
face_cascade = cv2.CascadeClassifier('C:/Users/Umar/Desktop/CV2/HaarCascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/Umar/Desktop/CV2/HaarCascade/haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('C:/Users/DarkLord/Downloads/Opencv_python_fundamentals-master/Opencv_python_fundamentals-master/haarcascades/haarcascade_smile.xml')
#FACE & EYE DETECTION FROM WEBCAM
cap=cv2.VideoCapture(0)# Capturing video from the camera

fourcc = cv2.VideoWriter_fourcc(*'XVID') #The fourcc is used to mention the video codec for encoding the video
#The VideoWriter() accepts four parameters.These are as follows:
    # 1) The output file name along with the format in which you want to save the file
    # 2) The video codec or fourcc.
    # 3) Frames per second which is used to mention how fast or slow your video should play along with the frame size that is the size of the window
    # 4) isColor flag. If it is True, encoder expect color frame, otherwise it works with grayscale frame.
out = cv2.VideoWriter('C:/Users/Umar/Desktop/CV2/Results/facedetection_output1.avi',fourcc, 20.0, (int(cap.get(3)),(int(cap.get(4)))))


# Step 3: Displaying it to the user
while(cap.isOpened()):
    ret, frame=cap.read()
    #Step 4: Perform Color Inversion on the frames
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)#This will convert BGR to RGB
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #prevent GUI crash
    cv2.startWindowThread()
    #for writing the frame in place where you want to save video 
    out.write(frame)
    #Step 5:- Displaying it to the user
    cv2.imshow("Face and Eye detection",frame)
    #Step 6: Checking for user's interaction  with keyboard
    if cv2.waitKey(25) & 0xFF==ord('a'):#Exit whenever user press the a key on the keyboard
        break
#Step 7: Releasing everything after the job is done
cap.release()
cv2.destroyAllWindows()

