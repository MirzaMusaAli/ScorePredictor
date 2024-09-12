import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor

# Load the pre-trained model pipeline C:\Users\mirza\Contacts\Score-predictor
pipe = pickle.load(open(r"C:\\Users\\mirza\\Contacts\\Score-predictor\\pipe.pkl", 'rb'))

# List of teams and cities
teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 'England', 'West Indies',
         'Afghanistan', 'Pakistan', 'Sri Lanka']

cities = ['Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 'London', 'Pallekele',
          'Barbados', 'Sydney', 'Melbourne', 'Durban', 'St Lucia', 'Wellington', 'Lauderhill',
          'Hamilton', 'Centurion', 'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton',
          'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 'Nagpur', 'Chandigarh',
          'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff', 'Christchurch', 'Trinidad']

# Streamlit app title
st.markdown('<h1 style="color:white;">T20 Cricket Score Predictor</h1>', unsafe_allow_html=True)

# Injecting CSS to add a background image and set text colors to white
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://png.pngtree.com/background/20240113/original/pngtree-d-illustration-of-a-cricket-stadium-with-a-front-view-and-picture-image_7253495.jpg");
        background-size: cover;
    }
    .stAlert {
        color: white;
        font-size: 25px;
    }
    .css-1d391kg p, .css-1d391kg h1, .css-1d391kg label, .css-1d391kg .st-bd {
        color: white;
    }
    .stNumberInput label, .stSelectbox label {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# User input columns
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

city = st.selectbox('Select city', sorted(cities))

col3, col4, col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score', min_value=0)
with col4:
    overs = st.number_input('Overs done (works for over > 5)', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets left', min_value=0, max_value=10, step=1)

last_five = st.number_input('Runs scored in last 5 overs', min_value=0)

# Prediction button
if st.button('Predict Score'):
    # Validation check for team selection
    if batting_team == bowling_team:
        st.markdown('<div style="color: #B22222; font-size:24px; font-weight: bold;">Invalid selection: Batting team and Bowling team cannot be the same.</div>', unsafe_allow_html=True)
    else:
        # Calculate additional inputs
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        crr = current_score / overs if overs > 0 else 0

        # Create input dataframe
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'current_score': [current_score],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left],
            'crr': [crr],
            'last_five': [last_five]
        })

        # Predict the score
        result = pipe.predict(input_df)
        st.markdown(f'<h2 style="color:white;">Predicted Score - {int(result[0])}</h2>', unsafe_allow_html=True)
