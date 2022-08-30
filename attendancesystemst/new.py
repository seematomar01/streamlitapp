import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import cv2
import numpy as np
from PIL import Image as im

db_name = 'face_rec_attendence1'
# datetime object containing current date and time
def employee_creation(serial):
    data_base = db_name
    serial = 'ID_' + str(serial)
    con = sqlite3.connect(f'{data_base}.db')
    cur = con.cursor()
    now = datetime.now()
    time = now.strftime("%d/%m/%Y %H:%M:%S")

    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    listof_table = (cur.fetchall())
    

    z = []
    for s in listof_table:
        z.append(s[0])

    if serial not in z:
       

        cur.execute(f"CREATE TABLE {serial}(Status text,time TIMESTAMP, image_sting text)")

        params = ('INITIAL',time, 'INITIAL')



        cur.execute(f"INSERT INTO {serial} VALUES (?,?, ?)", params)


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
    print(listof_table)
    df = employee_fetch(serial)
    if df['Status'].iloc[-1]!=status:

       
        params = (status,time, image_string)



        cur.execute(f"INSERT INTO {seriall} VALUES (?,?, ?)", params)
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




st.session_state['loggedIn'] = False
st.session_state['ADD'] = False
#if st.button("ADD_USER"):                                
#employee_seriel_no=st.number_input('Employee Seriel no')
#st.write('Employee Seriel no ',employee_seriel_no)
#img_file_buffer = st.file_uploader("Upload Image of Employee", type=["png","jpg","jpeg"])

def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")
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
                    st.write('Employee Seriel no ',employee_seriel_no)
                    img_file_buffer = st.file_uploader("Upload Image of Employee", type=["png","jpg","jpeg"])
                    if img_file_buffer is not None:
                        bytes_data = img_file_buffer.getvalue()
                        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)  
                        #cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                        img = np.array(cv2_img)
                        cv2.imwrite(f'employee images/{employee_seriel_no}.jpg', img)
                        employee_creation(employee_seriel_no)
                        st.image(img)




                if st.checkbox('check attendence'):
                    st.session_state['In'] = True
                    #employee_seriel_no  = st.number_input('Employee Seriel no')
                    if st.button('Show attendence'): 
                        df = employee_fetch(employee_seriel_no)
                        st.dataframe(df)

def page2():
    st.markdown("# Page 2 ❄️")
    st.sidebar.markdown("# Page 2 ❄️")
    if st.button("ADD_USER"):                                
        employee_seriel_no=st.number_input('Employee Seriel no')
        st.write('Employee Seriel no ',employee_seriel_no)
        img_file_buffer = st.file_uploader("Upload Image of Employee", type=["png","jpg","jpeg"])

def page3():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")

page_names_to_funcs = {
    "Main Page": main_page,
    "Page 2": page2,
    "Page 3": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()




# def Admin():

#     if st.session_state['loggedIn'] == False:
#         user = st.text_input('Username')
#         passwd = st.text_input('Password',type='password')
#         if st.button('Login') :


#             if user == 'b' and passwd == 'b' :
#                 st.session_state['loggedIn'] = True

#                 st.success("Logged In as {}".format(user))


#                 # Tasks For Only Logged In Users
                
#                 if st.button('check attendence'):
#                     st.session_state['In'] = True
#                     #employee_seriel_no  = st.number_input('Employee Seriel no')
#                     if st.button('Show attendence'): 
#                         df = employee_fetch(employee_seriel_no)
#                         st.dataframe(df)
                        
# Admin()

