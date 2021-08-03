import cv2
from random import randrange

trained_face_data= cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

webcam = cv2.VideoCapture(0)
while True:
  successful_frame_read, frame = webcam.read()
  #must be converted to grayscale
  grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  #detect faces
  face_coordinates = trained_face_data.detectMultiScale((grayscaled_img))
  #draw rectangle around the faces
  for (x,y,w,h) in face_coordinates:
    cv2.rectangle(frame,(x,y),(x+w, y+h), (randrange(256), randrange(256), randrange(256)), 3)
  cv2.imshow('myImg', frame)
  cv2.waitKey(2)

  #
  # #stop on keypres
  # if key==81 or key==113:
  #   break


print ('code completed')
webcam.release()