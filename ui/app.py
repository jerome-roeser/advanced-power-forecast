import datetime
import requests

from streamlit_folium import folium_static
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


'''
# Advanced Power Prediction
(v0.1)
'''

# create a sidebar in order to take user inputs
st.sidebar.markdown(f"""
    # User Input
    """)

prediction_date = st.sidebar.date_input(
                            label='Power prediction date',
                            value=datetime.date(2010, 1, 1),
                            min_value=datetime.date(1980, 1, 10),
                            max_value=datetime.date(2020, 1, 1),
                            )
predicition_time = st.sidebar.time_input(
                            label='Power prediction time',
                            value=datetime.time(0, 00),
                            step=3600)
input_prediction_date = f"{prediction_date} {predicition_time}"
# st.sidebar.write(input_prediction_date)


locations = st.sidebar.expander("Available locations")
days_to_display = st.sidebar.slider('Select the number of past data to display', 1, 10, 5)

input_prediction_date = f"{prediction_date} {predicition_time}"


location = locations.radio("Locations", ["Berlin - Tempelhof", "Berlin - Tegel", "Berlin - Schönefeld"])


# make api call
base_url = "http://127.0.0.1:8000"

# endpoints
endpoint_data = "/extract_data"
endpoint_baseline = "/predict/previous_value"
endpoint_model = "/predict"

# urls
url_data = f"{base_url}{endpoint_data}"
url_baseline = f"{base_url}{endpoint_baseline}"
url_model = f"{base_url}{endpoint_model}"

params = {
    'input_date':input_prediction_date
    }

# responses
response_data = requests.get(url_data, params=params).json()
data_df = pd.DataFrame(response_data)

response_baseline = requests.get(url_baseline, params=params).json()
baseline_df = pd.DataFrame(response_baseline)

# response_model = requests.get(url_model, params=params).json()
# model_df = pd.DataFrame(response_model)


# Main Panel
# Write name of chosen location
st.write(f"**Chosen location:** :red[{location}]")


# Graph for PV data
hours_to_display = 24 * days_to_display

# set-up 3 DatFrames according to input date and type of model
X = data_df.iloc[240-hours_to_display:-24,:]
y = data_df.iloc[-24:,:]
y_baseline = baseline_df.set_index(np.arange(240,240+24))
# y_model = model_df.set_index(np.arange(240,240+24))

fig, ax = plt.subplots()
ax.plot(X.get('electricity'), label='current production data')
ax.plot(y.get('electricity'), label='true data')
ax.plot(y_baseline.get(input_prediction_date), label='baseline data_API')
ax.legend()
plt.ylim(0,1)
plt.xlim(0,265)
st.pyplot(fig)


# Metrics
mean_training = X.get('electricity').mean()
mean_predicted = y_model.get(input_prediction_date).mean()
mean_diff = mean_predicted - mean_training

# Trick to use 4 columns to display the metrics centered below graph
col1, col2, col3, col4 = st.columns(4)
col2.metric("Training", round(mean_training,3), "")
col3.metric("Predicted", round(mean_predicted,3), round(mean_diff,3))


# Map with the location
coordinates = {
            'Berlin - Tempelhof':{'lat':52.4821,'lon':13.3892},
            'Berlin - Tegel':{'lat':52.5541,'lon':13.2931},
            'Berlin - Schönefeld':{'lat':52.3733,'lon':5064},
               }

map =folium.Map(
    location=[coordinates[location]['lat'],
              coordinates[location]['lon']],
    zoom_start=13)
folium.Marker([coordinates[location]['lat'], coordinates[location]['lon']],
              popup=location,
              icon=folium.Icon(color='red')).add_to(map)
folium_static(map)
