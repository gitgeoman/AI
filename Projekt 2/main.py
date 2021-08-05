import cv2
from random import randrange

#our input data
video=cv2.VideoCapture('Tesla Autopilot Dashcam Compilation 2018 Version.mp4')

#pre-trained classifier
classifier_file = 'cars.xml'
car_tracker = cv2.CascadeClassifier(classifier_file)

#running forever in loop
while True:
    (read_successful, frame) = video.read()
    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    print (cars)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 3)
    cv2.imshow('car detector',frame)
    cv2.waitKey(1)
print ('code completed')