import datetime
import requests
import streamlit as st


"""
Can we make a general description of th escript here ?
"""

'''
# Advanced Power Prediction
(v0.1)
'''

# create a sidebar in order to take user inputs
st.sidebar.markdown(f"""
    # User Input
    """)

prediction_date = st.sidebar.date_input('Power prediction date')
predicition_time = st.sidebar.date_input('Power prediction time')
expander = st.sidebar.expander("Available locations")

location = expander.radio("Locations", ["Berlin - Tempelhof"])


# make api call
# url = 'our-api-url_local'

# params= {'requests':'params'}

# repsonse = requests.get(url)
