import streamlit as st
from streamlit_option_menu import option_menu
from smartadashboard import home_page, visual_insights, sentiment_prediction, contact_us

# Set up the page configuration and any other initialization here
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")

# Sidebar menu using option_menu
with st.sidebar:
    menu = option_menu("Main Menu", ["Home", "Visual Insights", "Sentiment Prediction", "Contact Us"],
                       icons=["house", "bar-chart", "emoji-neutral", "envelope"],
                       menu_icon="menu-app-fill", default_index=0)

if menu == "Home":
    home_page()
elif menu == "Visual Insights":
    visual_insights()
elif menu == "Sentiment Prediction":
    sentiment_prediction()
elif menu == "Contact Us":
    contact_us()
