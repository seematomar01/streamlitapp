#import SessionState
import streamlit as st


st.markdown("# Demonstrating use of Next button with Session State")


page_number = 0

last_page = 1

# Add a next button and a previous button

prev, _ ,next = st.beta_columns([1, 10, 1])

if next.button("Next"):
    page_number = 1
    st.write("one")
    
if prev.button("Previous"):
    page_number = 0
    st.write("zero")    

# Get start and end indices of the next page of the dataframe
#start_idx = page_number * N 
#end_idx = (1 + page_number) * N

# Index into the sub dataframe
#sub_df = data.iloc[start_idx:end_idx]
#st.write(sub_df)