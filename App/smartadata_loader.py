import pandas as pd
import streamlit as st

@st.cache_data
def load_dataset():
    """Load and cache the dataset for faster performance"""
    return pd.read_csv("Dataset/labeled_tweets.csv")

data = load_dataset()
