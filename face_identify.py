import cv2
import numpy
import pickle

#places a facial recognition cascade into the variable
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#uses the LBPH algorithm to initialize a recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

#defines the recognizer with the trained model
recognizer.read("training.yml")

labels = {}

#grabs the labels from face_training.py
with open("labels.pickle", "rb") as f:
    labels = pickle.load(f)
    new_labels = {v:k for k,v in labels.items()}

video_capture = cv2.VideoCapture(0)

while True:

    check, color_frame = video_capture.read()

    gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)

    #gaussian blue smoothens the image so shadows are not detected
    gray_frame = cv2.GaussianBlur(gray_frame,(21, 21), 0)

    faces = faceCascade.detectMultiScale(gray_frame, scaleFactor = 1.25, minNeighbors = 5)    

    for x, y, w, h in faces:
        cv2.rectangle(color_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        region_gray = gray_frame[y:y+h, x:x+w]
        region_color = color_frame[y:y+h, x:x+w]

        ID, confidence_lvl = recognizer.predict(region_gray)

        #face is only identified when the confidence of the recognizer is high enough
        if confidence_lvl < 85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = new_labels[ID]
            cv2.putText(color_frame, name, (x,y), font, 1, (255,255,255), 2, cv2.LINE_AA)
            
        
        # used to create more images for training
        # img_face = "isaac10.png"
        # cv2.imwrite(img_face, region_color)
    
    cv2.imshow("Video", color_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()
