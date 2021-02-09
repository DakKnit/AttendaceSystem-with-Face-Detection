import numpy as np
import cv2 as cv # to install this use pip install opencv-python
''' to install next library use pip install face_recognition if there is any error that means you not have the 
c++ windows development kit you should install visual studio c++ environment'''
import face_recognition
import os
from datetime import datetime

path = 'faces'
images = []
studentNames = []
myList = os.listdir(path)
print(myList)

for cls in myList:
    curImg = cv.imread(f'{path}/{cls}')
    images.append(curImg)
    studentNames.append(os.path.splitext(cls)[0])
print(studentNames)


def findPeoples(images):
    students = []
    for img in images:
        encode = face_recognition.face_encodings(img)[0]
        students.append(encode)
    return students


knownFaces = findPeoples(images)
print(len(knownFaces))

wbCam = cv.VideoCapture(0)


def mack_attendance(name):
    with open("Attendance.csv", "r+") as f:
        my_data = f.readlines()
        namelist = []
        print(my_data)
        for line in my_data:
            entry = line.split(",")
            namelist.append(entry[0])
        if name not in namelist:
            time = datetime.now().strftime('%H:%M:%S')
            f.writelines(f'\n{name},{time}')


while True:
    success, frame = wbCam.read()
    frameS = cv.resize(frame, (0, 0), None, 0.25, 0.25)
    detectFaces = face_recognition.face_locations(frameS)
    encodes = face_recognition.face_encodings(frameS, detectFaces)
    for faceEncode, FaceLoc in zip(encodes, detectFaces):
        matchs = face_recognition.compare_faces(knownFaces, faceEncode)
        dis = face_recognition.face_distance(knownFaces, faceEncode)
        matchIndex = np.argmin(dis)
        if dis < 0.5: 
            name = studentNames[matchIndex]
            (y1, x2, y2, x1) = FaceLoc
            (y1, x2, y2, x1) = (y1*4, x2*4, y2*4, x1*4)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.rectangle(frame, (x1, y2), (x2, y2+35), (0, 255, 0), cv.FILLED)
            cv.putText(frame, name.upper(), (x1+6, y2+30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            mack_attendance(name)
        else:
            (y1, x2, y2, x1) = FaceLoc
            (y1, x2, y2, x1) = (y1*4, x2*4, y2*4, x1*4)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.rectangle(frame, (x1, y2), (x2, y2+35), (0, 255, 0), cv.FILLED)
            cv.putText(frame, "UNKNOWN", (x1+6, y2+30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    cv.imshow("cam", frame)
    key = cv.waitKey(1)

    if key == 27 or key == 10:
        break

wbCam.release()
print("process ended")



