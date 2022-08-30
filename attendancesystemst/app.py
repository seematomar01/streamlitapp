
import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional
import os
import pandas as pd
from datetime import  datetime
from PIL import Image as im
import cv2

import numpy as np

import streamlit as st

import base64
db_name = 'face_rec_attendence1'
##
import face_recognition as face_rec
import cv2
import shutil
path = 'employee images'
employeeImg = []

employeeName = []
myList = os.listdir(path)
filename = 'click'
FRAME_WINDOW = st.image([])
FRAME_WINDOW1 = st.image([])
cam0 = cv2.VideoCapture(0) 
cam1 = cv2.VideoCapture(1) 

st.session_state['loggedIn'] = False

def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)



def findEncoding(images) :
    imgEncodings = []
    for img in images :
        img = resize(img, 0.50)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgEncodings.append(encodeimg)
    return imgEncodings
def MarkAttendence(name):
    with open('attendence.csv', 'r+') as f:
        myDatalist =  f.readlines()
        nameList = []
        for line in myDatalist :
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            timestr = now.strftime('%H:%M')
            f.writelines(f'\n{name}, {timestr}')
            statment = str('welcome to seasia' + name)

for cl in myList :
    curimg = cv2.imread(f'{path}/{cl}')
    employeeImg.append(curimg)
    employeeName.append(os.path.splitext(cl)[0])

EncodeList = findEncoding(employeeImg)

import pandas as pd
import sqlite3
from datetime import datetime

# datetime object containing current date and time
def employee_creation(serial):
    data_base = db_name
    serial = 'ID_' + str(serial)
    con = sqlite3.connect(f'{data_base}.db')
    cur = con.cursor()


    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    listof_table = (cur.fetchall())
    




       


    cur.execute(f"CREATE TABLE {serial}(Status text,time TIMESTAMP, image_sting text)")

    now = datetime.now()
    time = now.strftime("%d/%m/%Y %H:%M:%S")



    params = ('Initial',time, 'Initial_str')



    cur.execute(f"INSERT INTO {seriall} VALUES (?,?, ?)", params)

    con.commit()
    con.close()


#detect employee
def employee_attendance(serial,status,image_string):
    
    serial = int(serial)    
    
    data_base = db_name
    
    now = datetime.now()
    time = now.strftime("%d/%m/%Y %H:%M:%S")
   
    seriall = 'ID_' + str(serial)
    con = sqlite3.connect(f'{data_base}.db')
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    listof_table = (cur.fetchall())
    z = []
    for s in listof_table:
        z.append(s[0])



    df = employee_fetch(serial)
    print(df)
    if df['Status'].iloc[-1]!=status:

       
        params = (status,time, image_string)



        cur.execute(f"INSERT INTO {seriall} VALUES (?,?, ?)", params)
        st.warning(f'attendance {status} {seriall}')
    con.commit()
    con.close()


#fetch employee
def employee_fetch(serial):
    data_base = db_name
    serial = 'ID_' + str(serial)
    con = sqlite3.connect(f'{data_base}.db')
    cur = con.cursor()


    df = pd.read_sql_query(f'SELECT * FROM {serial}', con)
    print(df)
    return df




def main():
    st.header("Attendence")

    pages = {

        "Attendence ": employ_recog,        
        #"Admin ": Admin,       
    }
    page_titles = pages.keys()

    page_title = st.sidebar.selectbox(
        "Choose the app mode",
        page_titles,
    )
    st.subheader(page_title)

    page_func = pages[page_title]
    page_func()

    st.sidebar.markdown(
        """
---
    """,  # noqa: E501
        unsafe_allow_html=True,
    )







def emprec0(img):    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    facesInFrame = face_rec.face_locations(img)

    encodeFacesInFrame = face_rec.face_encodings(img, facesInFrame)


    for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame) :
        matches = face_rec.compare_faces(EncodeList, encodeFace)
        facedis = face_rec.face_distance(EncodeList, encodeFace)
        print(facedis)
        if min(facedis) < 0.5:
            matchIndex = np.argmin(facedis)

            print(matchIndex)


            name = employeeName[matchIndex].upper()
            data = im.fromarray(img)
            data.save("data.jpg")

            image = open('data.jpg', 'rb')
            image_read = image.read()
            bs4str = base64.b64encode(image_read)


            #bs4str = base64.b64encode(img)
#             y1, x2, y2, x1 = faceloc
#             y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)
#             cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            top, right, bottom, left = faceloc
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, name,  (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            name = name.replace(' ','') 
            employee_attendance(name,'IN',bs4str)   
    return img   


def emprec1(img):    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    facesInFrame = face_rec.face_locations(img)

    encodeFacesInFrame = face_rec.face_encodings(img, facesInFrame)


    for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame) :
        matches = face_rec.compare_faces(EncodeList, encodeFace)
        facedis = face_rec.face_distance(EncodeList, encodeFace)
        print(facedis)
        if min(facedis) < 0.5:
            matchIndex = np.argmin(facedis)

            print(matchIndex)


            name = employeeName[matchIndex].upper()
            data = im.fromarray(img)
            data.save("data.jpg")

            image = open('data.jpg', 'rb')
            image_read = image.read()
            bs4str = base64.b64encode(image_read)


            #bs4str = base64.b64encode(img)
#             y1, x2, y2, x1 = faceloc
#             y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)
#             cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            top, right, bottom, left = faceloc
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(img, name,  (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            name = name.replace(' ','') 
            employee_attendance(name,'OUT',bs4str)   
    return img   



        
def Admin():

    if st.session_state['loggedIn'] == False:
        user = st.text_input('Username')
        passwd = st.text_input('Password',type='password')
        if st.button('Login') :


            if user == 'b' and passwd == 'b' :
                st.session_state['loggedIn'] = True

                st.success("Logged In as {}".format(user))


                # Tasks For Only Logged In Users
                
                if st.button('check attendence'):
                    st.session_state['In'] = True
                    #employee_seriel_no  = st.number_input('Employee Seriel no')
                    if st.button('Show attendence'): 
                        df = employee_fetch(employee_seriel_no)
                        st.dataframe(df)

                if st.button('Add User'):
                    employee_seriel_no  = st.number_input('Employee Seriel no')
                    img_file_buffer = st.file_uploader("Upload Image of Employee", type=["png","jpg","jpeg"])


                    if st.button('Next'):
                        if img_file_buffer is not None:
                            bytes_data = img_file_buffer.getvalue()
                            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)  
                            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                            img = np.array(cv2_img)
                            #bs4str = base64.b64encode(img)
                            st.image(img)
                            if st.button('Register Employee'):
                                cv2.imwrite(f'employee images/{employee_seriel_no}.jpg', img)
                                employee_creation(employee_seriel_no)
                                
                                st.success(f"{employee_seriel_no} Registered")
           
    
    
    




    
def employ_recog():
        st.title("Webcam Application")
        run = st.checkbox('Run')
        while run:
            ret0, frame0 = cam0.read()  

            #print(frame)       
            img0 = emprec0(frame0)


            ret1, frame1 = cam1.read()  

            #print(frame)       
            img1 = emprec1(frame1)            
            FRAME_WINDOW.image(img0)
            FRAME_WINDOW1.image(img1)












if __name__ == "__main__":
    import os



    main()
