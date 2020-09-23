import os
import numpy
import cv2
import pickle
from PIL import Image

#places a facial recognition cascade into the variable
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#uses the LBPH algorithm to initialize a recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

directory = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(directory, "face_images")

label_ID = 0
label_IDs = {}
X_train = []
labels = []

for root, dirs, files in os.walk(image_dir):
    for file in files:

        #finds the image files and saves the name associated with the file in label
        if file.endswith("png") or files.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()

            if not label in label_IDs:
                label_IDs[label] = label_ID
                label_ID += 1
            ID = label_IDs[label]
            
            #converts the image to gray
            pillow_image = Image.open(path).convert("L")

            resized = pillow_image.resize((500,500), Image.ANTIALIAS)

            #converts the image into an array of numbers based on the pixels
            image_array = numpy.array(resized, "uint8")

            faces = faceCascade.detectMultiScale(image_array, scaleFactor = 1.25, minNeighbors = 5)

            for (x,y,w,h) in faces:
                region = image_array[y:y+h, x:x+w]
                X_train.append(region)
                labels.append(ID)

#saves the labels into the file to be used in face_identify.py
with open("labels.pickle", 'wb') as f:
    pickle.dump(label_IDs, f)

#trains the model with the inputted images 
recognizer.train(X_train, numpy.array(labels))
recognizer.save("training.yml")
