import urllib.request
from pathlib import Path
from typing import List, NamedTuple, Optional
import os
import pandas as pd
from datetime import  datetime,date
today = date.today()
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
import numpy as np


def empdf(eid,name,dep,df):
    det = {'ID':eid,'Name':name,'Department':dep}

    df['time']= pd.to_datetime(df['time'])   
    datelist = list(df['time'].dt.date.unique())
    df['Date'] = df['time'].dt.date
    empdetat = pd.DataFrame(columns = ['ID','Name','Department','Date','IN','OUT','Total_working'])

    for da in datelist:
        dx = df[(df['Date'] == da)]
        date = da
        print(dx)
        intime = np.NaN
        outtime = np.NaN    
        total_work = np.NaN
        if len(dx) >1:
            dfg = work(dx)           
   

            
            intime = dfg['time_in'].iloc[0]
            outtime = dfg['time_out'].iloc[-1]
            
            print(type(outtime))
            if (type(outtime)== type(df['time'].iloc[0])) and (type(intime)== type(df['time'].iloc[0])):
                total_work =   (outtime-intime).total_seconds()/3600
            else:
                total_work =   outtime


        det['Date'] = date   
        det['IN'] = intime 
        det['OUT'] = outtime
        det['Total_working'] = total_work
        empdetat = empdetat.append(det, ignore_index=True)
    return empdetat

def work(df):

    df['Status_OUT'] = df['Status'] 
    df['time_out'] = df['time'] 
    df['Status_OUT'] = df['Status_OUT'].shift(-1)
    df['time_out'] = df['time_out'].shift(-1)
    df = df[((df['Status']  == 'IN') & (df['Status_OUT']  == 'OUT')) | (df['Status']  == 'IN')]
    df['time_out'] = pd.to_datetime(df['time_out'])
    df['time_in'] = pd.to_datetime(df['time'])
    df['Working hours'] = (df['time_out']-df['time']).dt.total_seconds()/3600
    df = df[['time_in','time_out','Working hours']]
    return df
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
def employee_creation(seriall,name,department,img):
    data_base = 'Employee_details'
    serial = 'ID_' + str(seriall)
    con = sqlite3.connect(f'{data_base}.db')
    now = datetime.now()
    time = now.strftime("%d/%m/%Y %H:%M:%S")
    cur = con.cursor()


    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    listof_table = (cur.fetchall())
    
    
    z = []
    for s in listof_table:
        z.append(s[0])

    if serial not in z:

    
        cur.execute(f"CREATE TABLE {serial}(Status text,time TIMESTAMP, image_sting text)")
        params = ('Initial',time, 'Initial_str')



        cur.execute(f"INSERT INTO {serial} VALUES (?,?, ?)", params)
        cur.execute(f"INSERT INTO Employee_detail VALUES (?,?, ?)", (seriall,name,department))
        con.commit()
        con.close()
        cv2.imwrite(f'employee images/{seriall}.jpg', img)
    else:
        st.write('Employee already exists')
#detect employee
def employee_attendance(serial,status,image_string):
    
    serial = int(serial)    
    
    data_base = 'Employee_details'
    
    now = datetime.now()
    time = now.strftime("%d/%m/%Y %H:%M:%S")
   
    seriall = 'ID_' + str(serial)
    con = sqlite3.connect(f'{data_base}.db')
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    listof_table = (cur.fetchall())
    df = employee_fetch(serial)
    if df['Status'].iloc[-1]!=status:

        print(df)
        if df['Status'].iloc[-1]!=status:
            params = (status,time, image_string)



            cur.execute(f"INSERT INTO {seriall} VALUES (?,?, ?)", params)
            st.warning(f'attendance {status} {seriall}')
        con.commit()
        con.close()


#fetch employee
def employee_fetch(serial):
    data_base = 'Employee_details'
    serial = 'ID_' + str(serial)
    con = sqlite3.connect(f'{data_base}.db')
    cur = con.cursor()


    df = pd.read_sql_query(f'SELECT * FROM {serial}', con)
    print(df)
    return df

st.session_state['loggedIn'] = False
st.session_state['ADD'] = False


#= 'Employee_details'
#con = sqlite3.connect(f'{data_base}.db')
#now = datetime.now()
#time = now.strftime("%d/%m/%Y %H:%M:%S")
#cur = con.cursor()
#cur.execute(f"CREATE TABLE Employee_detail(Emp_ID int,Emp_Name text, Department text)")



def employee_details_fetch(serial = None,name= None,department = None):

    db_name2 = 'Employee_details'
    print(db_name2)
    data_base = db_name2
    t = 'ID_' + str(serial)
    con = sqlite3.connect(f'{data_base}.db')
    cur = con.cursor()
    result_df = pd.DataFrame()
    df = pd.read_sql_query(f'SELECT * FROM Employee_detail', con)

    con.commit()
    con.close()    
    print(serial)
    print(df)
    if serial == '':
       serial = None

    if name == '':
       name = None

    if department == 'select department':
       department = None

    if (serial == None) and  (name == None)  and (department == None):
        print('bhai ismei phuncha hu')
        #result_df = df.loc[(df['Emp_ID'] == (serial)) & (df['Emp_Name'] == name)  & (df['Department'] == department)] 
        print(result_df)
        

    if (serial == None) and  (name == None)  and (department != None):
        result_df = df.loc[df['Department'] == department] 
        print(result_df)


    
    if (serial == None) and  (name != None)  and (department == None):
        result_df = df.loc[(df['Emp_Name'] == name) ] 
        print(result_df)

    if (serial == None) and (name != None) and (department != None):
        result_df = df.loc[ (df['Emp_Name'] == name) & (df['Department'] == department)]

    if (serial != None) and (name == None) and (department == None):
        result_df = df.loc[df['Emp_ID'] == int(serial)] 




    if (serial != None) and (name == None)  and (department != None):
        result_df = df.loc[(df['Emp_ID'] == int(serial)) & (df['Department'] == department)]    

    if (serial != None) and (name != None)  and (department == None):
        result_df = df.loc[(df['Emp_ID'] == int(serial)) & (df['Emp_Name'] == name) ]   

    if (serial != None) and  (name != None)  and (department != None):
        result_df = df.loc[(df['Emp_ID'] == int(serial)) & (df['Emp_Name'] == name)  & (df['Department'] == department)] 
        print(result_df) 
     
        
    return  result_df    
        
        




def main():
    image = im.open('seasia-infotech-largex5-logo.png')

    st.image(image, caption='Welcome to Seasia')

    st.header("Moogle Labs Attendance System")
    


    

    pages = {

        "Streaming ": employ_recog,        
        "Admin ": Admin,
        "Attendance": Check_attendance
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
        print(EncodeList,encodeFace)
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
        if st.checkbox('Login') :


            if user == 'b' and passwd == 'b' :
                st.session_state['loggedIn'] = True

                st.success("Logged In as {}".format(user))
                
                # Tasks For Only Logged In Users
                if st.checkbox("ADD_USER"):  
                    st.session_state['ADD'] = True                              
                    employee_seriel_no=st.text_input('Employee Seriel no')
                    name =st.text_input('Name')

                    option = st.selectbox('choose department',('select department','Machine Learning', 'Design', 'Digital marketing'))

                    department=option
                    st.write('Employee Seriel no ',employee_seriel_no)
                    img_file_buffer = st.file_uploader("Upload Image of Employee", type=["png","jpg","jpeg"])
                    if img_file_buffer is not None:
                        bytes_data = img_file_buffer.getvalue()
                        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)  
                        #cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                        img = np.array(cv2_img)

                        if st.checkbox("ADD_USER_TO_DB"):  
                            employee_creation(employee_seriel_no,name,department,img)
                            st.image(img)

    
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

        
def Check_attendance():
    if st.session_state['loggedIn'] == False:
        user = st.text_input('Username')
        passwd = st.text_input('Password',type='password')
        if st.checkbox('Login') :
            if user == 'b' and passwd == 'b' :
                st.session_state['loggedIn'] = True
                st.success("Logged In as {}".format(user))
                if st.checkbox('check attendence'):
                    st.session_state['In'] = True
        

                    Emp_ID  = st.text_input('Employee Seriel no')
                    Emp_name = st.text_input('Employee_name')



                    option = st.selectbox('choose department',('select department','Machine Learning', 'Design', 'Digital marketing'))

                    Department = option
                    print(Emp_ID,Emp_name,Department)
                    if st.checkbox('show attendence'):

                        start_date = st.date_input('Start date', today)
                        end_date = st.date_input('End date', today)

                        dfr = pd.DataFrame()

                        df =  employee_details_fetch(serial = Emp_ID,name = Emp_name,department = Department)
                        print(df) 
                        if  len(df) >0:
                            daka = pd.DataFrame()  
                            empids = list(df['Emp_ID'].unique())
                            print(empids)
                            for e in empids:
                                #st.write(f"{e} {employee_details_fetch(serial = e)['Emp_Name'].iloc[0]} {employee_details_fetch(serial = e)['Department'].iloc[0]}")
                                emp_name = employee_details_fetch(serial = e)['Emp_Name'].iloc[0]
                                dep = employee_details_fetch(serial = e)['Department'].iloc[0]                                
                                dafa = employee_fetch(e)


                                #st.dataframe(work(dafa[['Status','time']]))
                                #daka = daka.append(employee_fetch(e))
                                
                                dafa['time'] = pd.to_datetime(dafa['time'])
                                   

                                mask = (dafa['time'].dt.date <= end_date ) & (dafa['time'].dt.date >= start_date )
                                dafa = dafa.loc[mask]        

                                
                            

                                dafa = (dafa[['Status','time']])

                                daka =        daka.append(empdf(e,emp_name,dep,dafa)) 

                                #dft = work(dafa)
                            st.dataframe(daka.reset_index( drop=True)) 
                            if st.checkbox('show working_hours'):
                                df3 = work(df2)

     
                                #df3 = df3.applymap(str)

                                st.dataframe(df3)    
                                #dfz = df3[['time_in','Working hours']]
                                #dfz['Working hours'] = dfz['Working hours'].astype(float)
                                #dfz.index = dfz['time_in']
                                #st.bar_chart(dfz)



                             

                        else :
                            st.warning('Kindly check details') 
    








if __name__ == "__main__":
    import os



    main()
