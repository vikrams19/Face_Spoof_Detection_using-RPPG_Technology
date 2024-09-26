import cv2
from cv2 import *
from keras.preprocessing.image import img_to_array
import os
import numpy as np
from keras.models import model_from_json
import face_recognition
from datetime import datetime
from datetime import date 
import mysql.connector as connecter
from PIL import ImageGrab
import jpype


con = connecter.connect(host='localhost',port='3306',user='root',password='root',database='mp_finalTest')
print(con)

path = 'project_faces05'
images = []
classNames = []
namelist=[]

myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         encode = face_recognition.face_encodings(img)[0]
         encodeList.append(encode)
    return encodeList

class testcall :
    def markAttendance(self,Name,Entry_Date, Entry_Time):
        query = " insert into  python_table values(null,'{}','{}','{}'); ".format(Name,Entry_Date,Entry_Time)
        
        query2 = "insert into surplus1 Select * from Python_table join main_table1 on main_table1.Name=Python_table.Name order by Python_table.id desc limit 1 ;"
       
        query3 = "insert into Test_gate1 select prn ,name2 , department ,Vaccination_status  , entry_date , Entry_time , Year , admision_year , id2 from surplus1 order by id2 desc limit 1 ;"
       
        cur = con.cursor()
        cur.execute(query)
        print(query)
        cur.execute(query2)
        
        cur.execute(query3)
        
        con.commit()

Date = date.today()
Time =datetime.now()    
runa = testcall()

encodeListKnown = findEncodings(images)
print('Encoding Complete')
namelist=[]


root_dir = os.getcwd()
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

json_file = open('finalyearproject_antispoofing_model_mobilenet1.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('finalyearproject_antispoofing_model_97-0.943158.h5')
print("Model loaded from disk")

video = cv2.VideoCapture(0)
while True:
    
    try:
        ret,frame = video.read()
        imgS = cv2.resize(frame,(0,0),None,0.25,0.25)
        gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
           matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
           faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
           matchIndex = np.argmin(faceDis)
           name = classNames[matchIndex].upper()
        
        
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:   
           
            face = frame[y-5:y+h+5,x-5:x+w+5]
            resized_face = cv2.resize(face,(160,160))
            resized_face = resized_face.astype("float") / 255.0
           
            resized_face = np.expand_dims(resized_face, axis=0)
           
            preds = model.predict(resized_face)[0]
            
            if preds> 0.5:
                label = 'spoof'
                cv2.putText(frame, label, (x,y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cv2.rectangle(frame, (x, y), (x+w,y+h),
                    (0, 0, 255), 2)
            else:

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                mean_intensity = np.mean(roi_gray)
                if mean_intensity > 150 :
                    label = 'spoof'
                    cv2.putText(frame, label, (x,y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    cv2.rectangle(frame, (x, y), (x+w,y+h),
                    (0, 0, 255), 2)


                if mean_intensity <= 150 :
                    cv2.putText(frame, str(mean_intensity), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if matches[matchIndex] and name not in namelist :
                    namelist.append(name)
                    runa.markAttendance(name,Date,Time)

                
               
        cv2.imshow('frame', frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    except Exception as e:
        print("error")
   
video.release()        
cv2.destroyAllWindows()