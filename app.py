import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
from PIL import Image
import requests
from io import BytesIO
from collections import defaultdict
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

# Import necessary libraries for preprocessing
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from contractions import fix

# Load the trained model and vectorizer
model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Load your preprocessed dataset
data = pd.read_csv("Dataset/labeled_tweets.csv")

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# TEXT PROCESSING

# List of custom stopwords to retain from removing, including "no" and "not"
default_stopwords = set(stopwords.words('english'))
custom_stopwords = default_stopwords - {'no', 'not', 'nor', 'none', 'never', 'neither', 'without', 'against',
                                        'but', 'however', 'though', 'although',
                                        'because', 'since', 'due to', 'with'}


# Words that should not be lemmatized
non_lemmatizable_words = {'iphone16', 'iphone16plus', 'iphone16pro', 'iphone16promax', "ios", 'iphone15', 'iphone15plus', 'iphone15pro', 'iphone15promax',
                          'samsunggalaxy23', 'samsunggalaxys23plus', 'samsunggalaxys23ultra', 'samsunggalaxy24', 'samsunggalaxys24plus', 'samsunggalaxys24ultra',
                          'ios17', 'ios18', 'dynamic island', 'a17bionic', 'a18chip', 'usb-c', 'lightning port', 'pro motion', 'ceramic shield',
                          'snapdragon', 'exynos', '120hz', 'amozed', 'one ui',
                          '5g', 'refresh rate', 'fast charging', 'screen size'
                          }

# Function to map POS tags to WordNet tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

# Function for text preprocessing
def preprocess_text(text):
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Expand contractions (assuming `fix()` function is defined elsewhere)
    text = fix(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove special characters and punctuation, but keep numbers and dots
    text = re.sub(r'[^a-zA-Z0-9.\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove redundant whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize and POS tag
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()

    # Enhanced: Edge case handling for empty text
    if not tokens:
        return ''  # Return an empty string if no valid tokens remain

    # Processing tokens with enhanced handling
    processed_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        if word.isalpha() and word not in custom_stopwords and word not in non_lemmatizable_words
        else word
        for word, tag in pos_tags
    ]

    return ' '.join(processed_tokens)

# Function to predict sentiment for new tweets with preprocessing
def predict_tweet_sentiment(tweet_text):
    cleaned_text = preprocess_text(tweet_text)
    tweet_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(tweet_vectorized)
    sentiment_mapping_reverse = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
    return sentiment_mapping_reverse[prediction[0]]

# Features to analyze
features = ['design', 'display', 'camera', 'performance', 'battery', 'software', 'price']

# Function to extract adjectives using POS tagging
def extract_adjectives(text, feature):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    adjectives = []
    for idx, (word, tag) in enumerate(pos_tags):
        if word == feature:
            # Look for adjectives in a window around the feature
            window = pos_tags[max(0, idx - 2):idx + 3]
            adjectives.extend([w for w, t in window if t == 'JJ'])
    return adjectives

# Function to analyze sentiment for specific features
def analyze_sentiment_with_adjectives(row, features):
    sentiment_results = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0, 'adjectives': []})
    for feature in features:
        if feature in row['cleaned_text']:
            # Use the pre-calculated compound score
            compound = row['compound']
            if compound > 0.05:
                sentiment_results[feature]['positive'] += 1
            elif compound < -0.05:
                sentiment_results[feature]['negative'] += 1
            else:
                sentiment_results[feature]['neutral'] += 1
            # Extract adjectives using POS tagging
            adjectives = extract_adjectives(row['cleaned_text'], feature)
            sentiment_results[feature]['adjectives'].extend(adjectives)
    return sentiment_results

# Apply the updated function to the DataFrame
data['sentiment_results'] = data.apply(lambda row: analyze_sentiment_with_adjectives(row, features), axis=1)

# Summarize results by smartphone model
feature_sentiment_summary = defaultdict(lambda: defaultdict(dict))

for smartphone_model, group in data.groupby('smartphone'):
    model_results = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0, 'adjectives': []})
    for sentiment in group['sentiment_results']:
        for feature, values in sentiment.items():
            for key in ['positive', 'negative', 'neutral']:
                model_results[feature][key] += values[key]
            model_results[feature]['adjectives'].extend(values['adjectives'])
    feature_sentiment_summary[smartphone_model] = model_results

# Page configuration
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide", initial_sidebar_state="expanded")

# Define color palette for light and dark themes
# Define color palettes
THEMES = {
    "light": {
        "background": "#F9F9F9",
        "text": "#333333",
        "accent": "#FF6F61"
    },
    "dark": {
        "background": "#1E1E1E",
        "text": "#FFFFFF",
        "accent": "#FF6F61"
    }
}

def apply_theme(theme):
    st.markdown(f"""<style>
        .main {{
            background-color: {theme['background']};
            color: {theme['text']};
        }}
        .stButton > button {{
            background-color: {theme['accent']};
            color: white;
        }}
    </style>""", unsafe_allow_html=True)

# Theme toggle
def toggle_theme():
    if "theme" not in st.session_state:
        st.session_state.theme = "light"
    if st.button("Toggle Theme"):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
    return THEMES[st.session_state.theme]

current_theme = toggle_theme()
apply_theme(current_theme)

logo = Image.open('Picture\\Stata-no-bg.png')
# Sidebar menu
with st.sidebar:
    menu = option_menu("Main Menu", ["Home", "Visual Insights", "Sentiment Prediction","Contact Us"],
                       icons=["house", "bar-chart", "emoji-neutral", "envelope"], menu_icon="menu-app-fill", default_index=0)

if menu == "Home":
    # Display logo and tagline
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(logo, width=600)  # Adjust size as needed
    with col2:
        st.markdown("")
        st.title("\U0001F4C8 SMARTA")
        st.markdown("### **Smart Thinking for Smarter Insights**")
        st.info("*Turning User Sentiments into Smarter Decisions.*")
    
    # Create tabs
    tabs = st.tabs(["About Project", "Smartphones", "Dataset"])
    
    # Tab 1: About Project
    with tabs[0]:
        st.header("Welcome to SMARTA!")
        st.markdown(
            """
            ### About SMARTA
            SMARTA is a data-driven sentiment analysis dashboard designed for smartphone companies seeking 
            to gain deeper insights into public sentiment and feature-based feedback for their flagship models, 
            specifically Apple and Samsung smartphones. This tool leverages 
            **Natural Language Processing (NLP)** and **Machine Learning (ML)** to analyze social media discussions on X (formerly Twitter), 
            providing actionable insights to support **product improvement** and **strategic decision-making**.
            """
            
        )
        st.success("**Objective:** To provide actionable insights to support product improvement and strategic decision-making for smartphone company specifically Apple and Samsung.")

        # What SMARTA Delivers
        st.header("📈 What SMARTA Delivers:")
        st.markdown("""
        - **📊 Public Sentiment Trends:** Understand how users feel about your smartphones (Positive, Neutral, Negative).  
        - **🔍 Feature-Based Sentiment Analysis:** Evaluate **7 critical features**: **Design, Display, Camera, Battery, Software, Performance, and Price.**  
        - **📈 Data-Driven Insights:** Transform complex social media data into **clear insights** for product teams and marketers.  
        - **⚡ Real-Time Sentiment Prediction:** Instantly predict sentiment for new feedback or campaign launches.
        """)

        # Why SMARTA Matters for Smartphone Companies
        st.header("🎯 Why SMARTA Matters for Smartphone Companies:")
        st.markdown("""
        - **📊 Competitive Edge:** Identify how **iPhone 16, iPhone 15, Samsung S24, and Samsung S23** compare in public opinion.  
        - **📈 Product Improvement:** Uncover **user preferences** and **pain points** related to specific features for better product development.  
        - **🗣️ Marketing Insights:** Understand audience reactions to **product launches**, **design changes**, and **software updates**.  
        - **⚡ Faster Decisions:** Automate large-scale analysis, saving time on **manual review** of thousands of user opinions.
        """)

        # How SMARTA Works
        st.header("✅ How SMARTA Works:")
        st.markdown("""
        **1. Data Collection:**  
        - Crawls **X** data related to **iPhone 16, iPhone 15, Samsung S24, and Samsung S23.**  
        - Extracts real-time user reviews, comments, and reactions.  

        **2. Data Cleaning & Preprocessing:**  
        - Filters irrelevant data and expands contractions (e.g., *"can't"* → *"cannot"*).  
        - Uses **custom stopwords** to retain product-relevant terms and sentiment modifiers (e.g., *"no"* and *"not"*).  

        **3. Sentiment Analysis:**  
        - **SVM Machine Learning Model:** Classifies tweets as **Positive, Neutral, or Negative.**  
        - **Feature-Based Sentiment Analysis:** Evaluates **7 smartphone features** for specific feedback.  

        **4. Visualization & Insights:**  
        - **Interactive Charts:** Sentiment breakdowns, feature comparisons, and real-time predictions.  
        - **SMARTA Dashboard:** Clear summaries for product teams and marketers.
        """)

        # Real-World Use Case for Smartphone Companies
        st.header("🧑‍💼 Real-World Use Case for Smartphone Companies:")
        st.markdown("""
        **Scenario:**  
        - **Marketing Manager at Apple:** *"We want to know how users feel about our new iPhone's camera compared to Samsung's flagship."*  
        - **Product Developer at Samsung:** *"Which features are receiving the most criticism so we can improve our next update?"*  

        **SMARTA Solution:**  
        - Identifies **strengths** and **pain points** for both brands.  
        - Provides **feature-based sentiment trends** to prioritize improvements.
        """)

        # Conclusion
        st.success("""
        ### 🚀 Conclusion:
        **SMARTA empowers smartphone companies** with **data-backed insights** to stay ahead in the competitive market, **improve product quality**, and enhance **customer satisfaction** through **real-time social media analysis.**
        """)
        
        

    # Tab 2: Smartphones
    with tabs[1]:
        st.header("📱 Smartphone Models Analyzed")
        st.write("Learn more about the latest Apple and Samsung flagship models:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.header("")
            st.header("Samsung")
            # Samsung smartphones details
            Samsungs24 = [
                {
                    "name": "Samsung Galaxy S24 Ultra",
                    "images": {
                        "Titanium Gray":"Picture\\S24UTG-no-bg.png",
                        "Titanium Black":"Picture\\S24UTB-no-bg.png",
                        "Titanium Violet":"Picture\\S24UTV-no-bg.png",
                        "Titanium Yellow":"Picture\\S24TY-no-bg.png"
                    },
                    "description": """
                    - **Display**:6.8” Dynamic AMOLED 2X QHD+ (3120 x 1440), 1-120Hz, S Pen support
                    - **Rear Camera**: 
                        - 200 MP Wide
                        - 12 MP Ultra Wide
                        - 50 MP Telephoto (3x)
                        - 10 MP Telephoto (10x)
                    - **Front Camera**: 12 MP F2.2
                    - **Performance**: Snapdragon 8 Gen 3, 12GB RAM, 256GB/512GB/1TB Storage
                    - **Battery**:5000mAh, Wireless PowerShare
                    - **Price**: Starting at RM6,799.00
                    - **Release Date**: February 2024
                    """
                },
                {
                    "name": "Samsung Galaxy S24+",
                    "images": {
                        "Marble Gray":"Picture\\S24+MG-no-bg.png",
                        "Onyx Black":"Picture\\S24+OB-no-bg.png",
                        "Cobalt Violet":"Picture\\S24+CV-no-bg.png",
                        "Amber Yellow":"Picture\\S24+AY-no-bg.png"
                    },
                    "description": """
                    - **Display**: 6.7” Dynamic AMOLED 2X QHD+ (3120 × 1440), 1-120Hz
                    - **Rear Camera**: 50 MP Wide, 12 MP Ultra Wide, 10 MP Telephoto (3x)
                    - **Front Camera**: 12 MP F2.2
                    - **Performance**: Snapdragon 8 Gen 3, 12GB RAM, 256GB/512GB Storage
                    - **Battery**: 4900mAh, Wireless PowerShare
                    - **Price**: Starting at RM4,899.00
                    - **Release Date**: February 2024
                    """
                },
                {
                    "name": "Samsung Galaxy S24",
                    "images": {
                        "Marble Gray":"Picture\\S24+MG-no-bg.png",
                        "Onyx Black":"Picture\\S24+OB-no-bg.png",
                        "Cobalt Violet":"Picture\\S24+CV-no-bg.png",
                        "Amber Yellow":"Picture\\S24+AY-no-bg.png",
                    },
                    "description": """
                    - **Display**: 6.2” Dynamic AMOLED 2X FHD+ (2340 x 1080), 1-120Hz
                    - **Rear Camera**: 50 MP Wide, 12 MP Ultra Wide, 10 MP Telephoto (3x)
                    - **Front Camera**: 12 MP F2.2
                    - **Performance**: Snapdragon 8 Gen 3, 8GB RAM, 128GB/256GB/512GB Storage
                    - **Battery**: 4000mAh, Wireless PowerShare
                    - **Price**: Starting at RM4,099.00
                    - **Release Date**: February 2024
                    """
                }
            ]

            # Display each smartphone with color selection
            for phone in Samsungs24:
                st.subheader(phone["name"])

                # Allow the user to select a color
                model_option = st.selectbox(f"Choose Models for {phone['name']}", list(phone["images"].keys()))

                # Load and display the selected image
                selected_image_path = phone["images"][model_option]
                img = Image.open(selected_image_path)
                st.image(img, width=200, caption=f"{phone['name']} - {model_option}")

                # Display the description in a styled box
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ccc; padding: 15px; border-radius: 5px; background-color: #f9f9f9; color: black;">
                        {phone['description']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write("")  # Spacer between phones

            # Samsung smartphones details
            SamsungS23 = [
                {
                    "name": "Samsung Galaxy S23 Ultra",
                    "images": {
                        "Cream":"Picture\\S23UC-no-bg.png",
                        "Phantom Black":"Picture\\S23UBP-no-bg.png",
                        "Green":"Picture\\S23UG-no-bg.png",
                        "Lavender":"Picture\\S23UL-no-bg.png"
                    },
                    "description": """
                    - **Display**: 6.8” Dynamic AMOLED 2X, 120Hz, HDR10+, 1200 nits (HBM), 1750 nits (peak)
                    - **Rear Camera**: 200 MP, f/1.7, 24mm (wide), 1/1.3", 0.6µm, multi-directional PDAF, OIS
                    10 MP, f/2.4, 70mm (telephoto), 1/3.52", 1.12µm, PDAF, OIS, 3x optical zoom
                    10 MP, f/4.9, 230mm (periscope telephoto), 1/3.52", 1.12µm, PDAF, OIS, 10x optical zoom
                    12 MP, f/2.2, 13mm, 120˚ (ultrawide), 1/2.55", 1.4µm, dual pixel PDAF, Super Steady video
                    - **Front Camera**: 12 MP, f/2.2, 26mm (wide), dual pixel PDAF
                    - **Performance**: Snapdragon 8 Gen 2, 12GB RAM, 256GB/512GB/1TB Storage
                    - **Battery**: 5000mAh, Wireless PowerShare
                    - **Price**: Starting at RM5,999.00
                    - **Release Date**: February 2023
                    """
                },
                {
                    "name": "Samsung Galaxy S23+",
                    "images": {
                        "Cream":"Picture\\S23C-no-bg.png",
                        "Phantom Black":"Picture\\S23BP-no-bg.png",
                        "Green":"Picture\\S23G-no-bg.png",
                        "Lavender":"Picture\\S23L-no-bg.png"
                    },
                    "description": """
                    - **Display**: 6.6” Dynamic AMOLED 2X, 120Hz, HDR10+, 1200 nits (HBM), 1750 nits (peak)
                    - **Rear Camera**: 50 MP, f/1.8, 24mm (wide), 1/1.56", 1.0µm, dual pixel PDAF, OIS
                    10 MP, f/2.4, 70mm (telephoto), 1/3.94", 1.0µm, PDAF, OIS, 3x optical zoom
                    12 MP, f/2.2, 13mm, 120˚ (ultrawide), 1/2.55", 1.4µm, Super Steady video
                    - **Front Camera**: 12 MP, f/2.2, 26mm (wide), dual pixel PDAF
                    - **Performance**: Snapdragon 8 Gen 2, 8GB RAM, 256GB/512GB Storage
                    - **Battery**: 4700mAh, Wireless PowerShare
                    - **Price**: Starting at RM4,499.00
                    - **Release Date**: February 2023
                    """
                },
                {
                    "name": "Samsung Galaxy S23",
                    "images": {
                        "Cream":"Picture\\S23C-no-bg.png",
                        "Phantom Black":"Picture\\S23BP-no-bg.png",
                        "Green":"Picture\\S23G-no-bg.png",
                        "Lavender":"Picture\\S23L-no-bg.png"
                    },
                    "description": """
                    - **Display**: 6.1” Dynamic AMOLED 2X, 120Hz, HDR10+, 1200 nits (HBM), 1750 nits (peak)
                    - **Rear Camera**: 50 MP, f/1.8, 24mm (wide), 1/1.56", 1.0µm, dual pixel PDAF, OIS
                    10 MP, f/2.4, 70mm (telephoto), 1/3.94", 1.0µm, PDAF, OIS, 3x optical zoom
                    12 MP, f/2.2, 13mm, 120˚ (ultrawide), 1/2.55", 1.4µm, Super Steady video
                    - **Front Camera**: 12 MP, f/2.2, 26mm (wide), dual pixel PDAF
                    - **Performance**: Snapdragon 8 Gen 2, 8GB RAM, 128GB/256GB/512GB Storage
                    - **Battery**: 3900mAh, Wireless PowerShare
                    - **Price**: Starting at RM3,999.00
                    - **Release Date**: February 2023
                    """
                }
            ]

            # Display each smartphone with color selection
            for phone in SamsungS23:
                st.subheader(phone["name"])

                # Allow the user to select a model
                model_option = st.selectbox(f"Choose Model for {phone['name']}", list(phone["images"].keys()))

                # Load and display the selected image
                selected_image_path = phone["images"][model_option]
                img = Image.open(selected_image_path)
                st.image(img, width=200, caption=f"{phone['name']} - {model_option}")

                # Display the description dynamically based on the selected model
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ccc; padding: 15px; border-radius: 5px; background-color: #f9f9f9; color: black;">
                        {phone['description']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write("")  # Spacer between phones

        with col2:
            st.header("")
            st.header("Apple")
            iPhone16 = [
                {
                    "name": "iPhone 16 Pro Max",
                    "images": {
                        "Desert Titanium":"Picture\\16DT-no-bg.png",
                        "Black Titanium":"Picture\\16BT-no-bg.png",
                        "Natural Titanium":"Picture\\16NT-no-bg.png",
                        "White Titanium":"Picture\\16WT-no-bg.png"
                    },
                    "description": """
                    - **Display**: 6.9 inches Super Retina XDR Display with ProMotion (2868 x 1320)
                    - **Rear Camera**: 48 MP main (f/1.78), 48 MP ultrawide (f/2.2), 12 MP telephoto (5x, f/2.8)
                    - **Front Camera**: 12 MP (f/1.9)
                    - **Performance**: A18 Pro Chip
                    - **Battery**: 4,685 mAh, up to 18:06 hours of video playback
                    - **Price**: Starting at RM5,999
                    - **Release Date**: October 2024
                    """
                },
                {
                    "name": "iPhone 16 Pro",
                    "images": {
                        "Desert Titanium":"Picture\\16DT-no-bg.png",
                        "Black Titanium":"Picture\\16BT-no-bg.png",
                        "Natural Titanium":"Picture\\16NT-no-bg.png",
                        "White Titanium":"Picture\\16WT-no-bg.png"
                    },
                    "description": """
                    - **Display**: 6.3 inches Super Retina XDR Display with ProMotion (2622 x 1206)
                    - **Rear Camera**: 48 MP main (f/1.78), 48 MP ultrawide (f/2.2), 12 MP telephoto (5x, f/2.8)
                    - **Front Camera**: 12 MP (f/1.9)
                    - **Performance**: A18 Pro Chip
                    - **Battery**: 3,582 mAh, up to 14:07 hours of video playback
                    - **Price**: Starting at RM5,499
                    - **Release Date**: October 2024
                    """
                },
                {
                    "name": "iPhone 16 Plus",
                    "images": {
                        "Ultramarine":"Picture\\16U-no-bg.png",
                        "Teal":"Picture\\16T-no-bg.png",
                        "White":"Picture\\16W-no-bg.png",
                        "Pink":"Picture\\16P-no-bg.png",
                        "Black":"Picture\\16BL-no-bg.png"
                    },
                    "description": """
                    - **Display**: 6.7 inches Super Retina XDR Display
                    - **Rear Camera**: 48 MP main (f/1.6) with 2x optical quality zoom, 12 MP ultrawide (f/2.2)
                    - **Front Camera**: 12 MP (f/1.9)
                    - **Performance**: A18 Chip
                    - **Battery**: 4,674 mAh, up to 16:29 hours of video playback
                    - **Price**: Starting at RM4,299
                    - **Release Date**: October 2024
                    """
                },
                {
                    "name": "iPhone 16",
                    "images": {
                        "Ultramarine":"Picture\\16U-no-bg.png",
                        "Teal":"Picture\\16T-no-bg.png",
                        "White":"Picture\\16W-no-bg.png",
                        "Pink":"Picture\\16P-no-bg.png",
                        "Black":"Picture\\16BL-no-bg.png"
                    },
                    "description": """
                    - **Display**: 6.1 inches Super Retina XDR Display
                    - **Rear Camera**: 48 MP main (f/1.6) with 2x optical quality zoom, 12 MP ultrawide (f/2.2)
                    - **Front Camera**: 12 MP (f/1.9)
                    - **Performance**: A18 Chip
                    - **Battery**: 3,561 mAh, up to 12:43 hours of video playback
                    - **Price**: Starting at RM3,799
                    - **Release Date**: October 2024
                    """
                }
                
                
                
            ]

            # Display each iPhone 16 model with color selection
            for phone in iPhone16:
                st.subheader(phone["name"])

                # Allow the user to select a model
                model_option = st.selectbox(f"Choose Model for {phone['name']}", list(phone["images"].keys()))

                # Load and display the selected image
                selected_image_path = phone["images"][model_option]
                img = Image.open(selected_image_path)
                st.image(img, width=200, caption=f"{phone['name']} - {model_option}")

                # Display the description dynamically based on the selected model
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ccc; padding: 15px; border-radius: 5px; background-color: #f9f9f9; color: black;">
                        {phone['description']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write("")  # Spacer between phones

            iPhone15 = [
                {
                    "name": "iPhone 15 Pro Max",
                    "images": {
                        "Desert Titanium":"Picture\\15BT-no-bg.png",
                        "Black Titanium":"Picture\\15BLT-no-bg.png",
                        "Natural Titanium":"Picture\\15NT-no-bg.png",
                        "White Titanium":"Picture\\15WT-no-bg.png"
                    },
                    "description": """
                    - **Display**: 6.7 inches Super Retina XDR Display with ProMotion
                    - **Rear Camera**: 48 MP main, 12 MP ultrawide, 12 MP 5x telephoto
                    - **Front Camera**: 12 MP
                    - **Performance**: A17 Pro Chip
                    - **Battery**: 4,422 mAh, up to 14:02 hours of video playback
                    - **Price**: Starting at RM5,499
                    - **Release Date**: September 2023
                    """
                },
                {
                    "name": "iPhone 15 Pro",
                    "images": {
                        "Desert Titanium":"Picture\\15BT-no-bg.png",
                        "Black Titanium":"Picture\\15BLT-no-bg.png",
                        "Natural Titanium":"Picture\\15NT-no-bg.png",
                        "White Titanium":"Picture\\15WT-no-bg.png"
                    },
                    "description": """
                    - **Display**: 6.1 inches Super Retina XDR Display with ProMotion
                    - **Rear Camera**: 48 MP main, 12 MP ultrawide, 12 MP 3x telephoto
                    - **Front Camera**: 12 MP
                    - **Performance**: A17 Pro Chip
                    - **Battery**: 3,274 mAh, up to 10:53 hours of video playback
                    - **Price**: Starting at RM5,199
                    - **Release Date**: September 2023
                    """
                },
                {
                    "name": "iPhone 15 Plus",
                    "images": {
                        "Pink":"Picture\\15P-no-bg.png",
                        "Green":"Picture\\15G-no-bg.png",
                        "Blue":"Picture\\15B-no-bg.png",
                        "Black":"Picture\\15BL-no-bg.png",
                        "Yellow":"Picture\\15Y-no-bg.png"
                    },
                    "description": """
                    - **Display**: 6.7 inches Super Retina XDR Display
                    - **Rear Camera**: 48 MP main, 12 MP ultrawide
                    - **Front Camera**: 12 MP
                    - **Performance**: A16 Bionic Chip
                    - **Battery**: 4,383 mAh, up to 14:14 hours of video playback
                    - **Price**: Starting at RM3,859
                    - **Release Date**: September 2023
                    """
                },
                {
                    "name": "iPhone 15",
                    "images": {
                        "Pink":"Picture\\15P-no-bg.png",
                        "Green":"Picture\\15G-no-bg.png",
                        "Blue":"Picture\\15B-no-bg.png",
                        "Black":"Picture\\15BL-no-bg.png",
                        "Yellow":"Picture\\15Y-no-bg.png"
                    },
                    "description": """
                    - **Display**: 6.1 inches Super Retina XDR Display
                    - **Rear Camera**: 48 MP main, 12 MP ultrawide
                    - **Front Camera**: 12 MP
                    - **Performance**: A16 Bionic Chip
                    - **Battery**: 3,349 mAh, up to 11:05 hours of video playback
                    - **Price**: Starting at RM3,299
                    - **Release Date**: September 2023
                    """
                }
            ]

            # Display each iPhone 15 model with color selection
            for phone in iPhone15:
                st.subheader(phone["name"])

                # Allow the user to select a model
                model_option = st.selectbox(f"Choose Model for {phone['name']}", list(phone["images"].keys()))

                # Load and display the selected image
                selected_image_path = phone["images"][model_option]
                img = Image.open(selected_image_path)
                st.image(img, width=200, caption=f"{phone['name']} - {model_option}")

                # Display the description dynamically based on the selected model
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ccc; padding: 15px; border-radius: 5px; background-color: #f9f9f9; color: black;">
                        {phone['description']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.write("")  # Spacer between phones

    # Tab 3: Dataset
 
    with tabs[2]:
        st.markdown("## 📊 Explore the Dataset")
        st.write("Here's an overview of the dataset used for sentiment analysis:")
        st.write("🗓️ Data Last Updated: **December 31, 2024**")
        st.write("The dataset is refreshed monthly to ensure up-to-date analysis.")
        st.write("")
        selected_year = st.selectbox("Select Year:", options=data['Year'].unique())
        selected_smartphone = st.selectbox("Select Smartphone Model:", options=data['smartphone'].unique())
        keyword_search = st.text_input("Search Keywords:")

        # Apply filters
        filtered_data = data[
            (data['Year'] == selected_year) &
            (data['smartphone'] == selected_smartphone) &
            (data['full_text'].str.contains(keyword_search, case=False))
        ]

        st.markdown("### 📌 **Filtered Results:**")
        st.dataframe(filtered_data[["full_text", "favorite_count", "retweet_count", "reply_count", "smartphone", "compound", "sentiment", "sentiment_results"]])
        st.write("### Dataset Variables:")
        st.markdown(
            """
            - **full_text**: The content of the tweet (used for sentiment analysis).
            - **favorite_count:** Number of likes the tweet received.  
            - **retweet_count:** Number of retweets the tweet received.  
            - **reply_count:** Number of replies on the tweet. 
            - **smartphone:** Brand of the smartphone.
            - **compound:** Sentiment score calculated using VADER. 
            - **sentiment**: The classified sentiment (Positive, Neutral, or Negative).
            - **sentiment_result**: Show the mentioned features with adjectives.
            """
        )


elif menu == "Visual Insights":

    # Top-level metrics
    st.subheader("Summary Metrics")
    num_tweets = len(data)
    positive_tweets = len(data[data["sentiment"] == "Positive"])
    negative_tweets = len(data[data["sentiment"] == "Negative"])
    neutral_tweets = len(data[data["sentiment"] == "Neutral"])

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.image("Picture\\Xlogo-no-bg.png", width=600)  # Adjust width as needed
    col2.metric("Total Tweets", num_tweets)
    col3.metric("Positive Sentiments 😊", positive_tweets)
    col4.metric("Negative Sentiments 😡", negative_tweets)
    col5.metric("Neutral Sentiment 😐", neutral_tweets)


    # Tabs for tweets, smartphones, and sentiments
    tabs = st.tabs(["Tweets", "Smartphones"])

    with tabs[0]:
        st.subheader("Tweet Insights")
        st.dataframe(data[["full_text", "favorite_count", "retweet_count", "reply_count"]])

        st.markdown("#### Tweet Activity by Month and Year")
        if "Year" in data.columns and "Month" in data.columns:
            year_month_counts = data.groupby(['Year', 'Month']).size().reset_index(name='tweet_count')

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=year_month_counts, x='Month', y='tweet_count', hue='Year', marker='o', ax=ax)
            ax.set_title("Tweet Activity by Month and Year")
            ax.set_xlabel("Month")
            ax.set_ylabel("Tweet Count")
            ax.legend(title="Year")
            st.pyplot(fig)
        else:
            st.error("Year or Month columns missing from the dataset.")

        st.markdown("#### Monthly Tweet Activity by Smartphone and Year")
        if "smartphone" in data.columns and "Year" in data.columns and "Month" in data.columns:
            tweets_by_smartphone = data.groupby(['smartphone', 'Year', 'Month']).size().reset_index(name='tweet_count')
            tweets_by_smartphone['Year-Month'] = tweets_by_smartphone['Year'].astype(str) + '-' + tweets_by_smartphone['Month'].astype(str)

            fig, ax = plt.subplots(figsize=(16, 8))
            sns.barplot(data=tweets_by_smartphone, x='Year-Month', y='tweet_count', hue='smartphone', ax=ax)
            ax.set_title("Monthly Tweet Activity by Smartphone and Year")
            ax.set_xlabel("Year-Month")
            ax.set_ylabel("Tweet Count")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend(title="Smartphone Model")
            st.pyplot(fig)
        else:
            st.error("Year, Month, or Smartphone columns missing from the dataset.")


        # ====== Additional Insights (Optional) =======
        st.markdown("### 📊 Additional Insights")
        st.write("Would you like to see:")
        st.markdown("### 🔥 Tweets with Highest Engagement")
        show_high_engagement_tweets = st.checkbox("Show Top Engaged Tweets")

        if show_high_engagement_tweets:
            st.markdown("### 📊 **Engagement Insights Summary**")

            # Identifying the most liked and retweeted tweet
            most_liked_tweet = data.loc[data['favorite_count'].idxmax()]
            most_retweeted_tweet = data.loc[data['retweet_count'].idxmax()]

            # Displaying metrics
            col1, col2 = st.columns(2)
            col1.metric("Most Liked Tweet", f"{most_liked_tweet['favorite_count']} Likes")
            col2.metric("Most Retweeted Tweet", f"{most_retweeted_tweet['retweet_count']} Retweets")

            # Display the most liked and retweeted tweet content
            st.markdown("**Most Liked Tweet:**")
            st.write(most_liked_tweet['full_text'])

            st.markdown("**Most Retweeted Tweet:**")
            st.write(most_retweeted_tweet['full_text'])

            # Displaying top 5 tweets sorted by engagement
            st.markdown("### 📌 **Top 5 Tweets with Highest Engagement:**")
            top_tweets = data.sort_values(by=['favorite_count', 'retweet_count', 'reply_count'], ascending=False).head(5)
            st.dataframe(top_tweets[['full_text', 'favorite_count', 'retweet_count', 'reply_count', 'smartphone', 'sentiment']])


        st.success("🎯 **End of Visual Insights Section!** Explore the data and make informed decisions.")

    with tabs[1]:
        # Sentiment Analysis Section
        st.subheader("📊 Sentiment Analysis")
        
        # Sentiment Distribution Pie Chart (with standardized colors)
        sentiment_counts = data['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig = px.pie(
            sentiment_counts,
            values='Count',
            names='Sentiment',
            title="Sentiment Distribution",
            hole=0.3,
            color='Sentiment',
            color_discrete_map={'Positive': 'green', 'Neutral': 'yellow', 'Negative': 'red'}
        )
        st.plotly_chart(fig)

        # Smartphone Sentiment Analysis Bar Chart (standardized colors)
        st.subheader("📱 Smartphone Sentiment Analysis")
        if "smartphone" in data.columns and "sentiment" in data.columns:
            smartphone_sentiment = data.groupby(['smartphone', 'sentiment']).size().unstack(fill_value=0)
            fig = px.bar(
                smartphone_sentiment,
                title="Smartphone Sentiment Analysis",
                barmode="stack",
                color_discrete_sequence=['red', 'yellow', 'green']  # Standardized colors
            )
            st.plotly_chart(fig)
        else:
            st.error("Smartphone or Sentiment columns missing from the dataset.")

        # Feature-Based Sentiment Analysis Section
        st.subheader("🔎 Feature-Based Sentiment Analysis")

        # User selection
        selected_feature = st.selectbox("Select Feature for Analysis:", options=features)
        selected_model = st.selectbox("Select Smartphone Model:", options=data['smartphone'].unique())

        # Filter Data for Feature and Smartphone Model
        feature_filtered_data = data[data['smartphone'] == selected_model]

        # Process Sentiment Data for Selected Feature
        sentiment_summary = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0, 'adjectives': []})
        feature_messages = {'positive': [], 'negative': [], 'neutral': []}

        for _, row in feature_filtered_data.iterrows():
            for feature, values in row['sentiment_results'].items():
                if feature == selected_feature:
                    for sentiment in ['positive', 'negative', 'neutral']:
                        sentiment_summary[feature][sentiment] += values[sentiment]
                        # Extract top messages based on compound score
                        if row['compound'] >= 0.34 and sentiment == 'positive':
                            feature_messages['positive'].append((row['compound'], row['full_text']))
                        elif -0.05 < row['compound'] < 0.34 and sentiment == 'neutral':
                            feature_messages['neutral'].append((row['compound'], row['full_text']))
                        elif row['compound'] <= -0.05 and sentiment == 'negative':
                            feature_messages['negative'].append((row['compound'], row['full_text']))
                    sentiment_summary[feature]['adjectives'].extend(values['adjectives'])

        # Extract Sentiment Counts for Visualization
        sentiment_counts = [sentiment_summary[selected_feature][sentiment] for sentiment in ['positive', 'negative', 'neutral']]
        sentiment_df = pd.DataFrame({"Sentiment": ['Positive', 'Negative', 'Neutral'], "Count": sentiment_counts})

        # Bar Chart for Sentiment Count by Feature
        fig = px.bar(
            sentiment_df,
            x='Sentiment',
            y='Count',
            color='Sentiment',
            title=f"Sentiment Count for {selected_feature} - {selected_model}",
            color_discrete_map={'Positive': 'green', 'Neutral': 'yellow', 'Negative': 'red'}
        )
        st.plotly_chart(fig)

        # **Summary Box**
        total_sentiment_count = sum(sentiment_counts)
        sentiment_ratios = {
            'Positive': sentiment_counts[0] / total_sentiment_count * 100 if total_sentiment_count > 0 else 0,
            'Neutral': sentiment_counts[2] / total_sentiment_count * 100 if total_sentiment_count > 0 else 0,
            'Negative': sentiment_counts[1] / total_sentiment_count * 100 if total_sentiment_count > 0 else 0,
        }
        top_adjectives = list(set(sentiment_summary[selected_feature]['adjectives']))

        # Extract Top 3 Messages for Each Sentiment
        def extract_top_messages(messages, sentiment):
            sorted_msgs = sorted(messages[sentiment], key=lambda x: x[0], reverse=(sentiment == 'positive'))
            return [f"{msg[1]} (Compound: {msg[0]:.2f})" for msg in sorted_msgs[:3]]

        top_positive_msgs = extract_top_messages(feature_messages, 'positive')
        top_neutral_msgs = extract_top_messages(feature_messages, 'neutral')
        top_negative_msgs = extract_top_messages(feature_messages, 'negative')

        # Display Summary
        st.markdown(f"""
        ### 📈 Sentiment Summary for {selected_feature} ({selected_model}):
        - **Total Sentiment Count:** {total_sentiment_count}
        - **Sentiment Ratios:** Positive: {sentiment_ratios['Positive']:.2f}%, Neutral: {sentiment_ratios['Neutral']:.2f}%, Negative: {sentiment_ratios['Negative']:.2f}%
        - **Top Adjectives:** {', '.join(top_adjectives)}

        ### 📝 Top Messages:
        **Positive Sentiments:**
        1. {top_positive_msgs[0] if top_positive_msgs else "N/A"}
        2. {top_positive_msgs[1] if len(top_positive_msgs) > 1 else "N/A"}
        3. {top_positive_msgs[2] if len(top_positive_msgs) > 2 else "N/A"}

        **Neutral Sentiments:**
        1. {top_neutral_msgs[0] if top_neutral_msgs else "N/A"}
        2. {top_neutral_msgs[1] if len(top_neutral_msgs) > 1 else "N/A"}
        3. {top_neutral_msgs[2] if len(top_neutral_msgs) > 2 else "N/A"}

        **Negative Sentiments:**
        1. {top_negative_msgs[0] if top_negative_msgs else "N/A"}
        2. {top_negative_msgs[1] if len(top_negative_msgs) > 1 else "N/A"}
        3. {top_negative_msgs[2] if len(top_negative_msgs) > 2 else "N/A"}
        """)


        # ====== Word Cloud Section =======
        st.header("☁️ Word Cloud")

        # Stopword Removal Toggle
        remove_stopwords = st.checkbox("Remove Stopwords for Clarity")

        # Generate Word Cloud with Hover Effects
        text = " ".join(data['cleaned_text'])
        if remove_stopwords:
            from nltk.corpus import stopwords
            stopwords_set = set(stopwords.words('english'))
            wordcloud = WordCloud(width=800, height=400, stopwords=stopwords_set).generate(text)
        else:
            wordcloud = WordCloud(width=800, height=400).generate(text)

        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        st.success("🎯 **End of Visual Insights Section!** Explore the data and make informed decisions.")

elif menu == "Sentiment Prediction":   
    st.markdown("### 🎭 Sentiment Prediction")
    st.info("Predict the sentiment of a tweet instantly! Enter the text below to see whether it's Positive, Neutral, or Negative.")

    # Text input for user tweet
    user_input = st.text_area("💬 Enter tweet text here:", placeholder="Type your tweet here...")

    # Button to trigger prediction
    if st.button("Analyze Sentiment"):
        # Check if input is provided
        if user_input.strip():
            # Predict sentiment using the defined function
            predicted_sentiment = predict_tweet_sentiment(user_input)

            # Display the result
            st.success(f"**Predicted Sentiment:** {predicted_sentiment}")

            # Add an emoji for a better user experience
            sentiment_emoji = {
                'Positive': "😊",
                'Neutral': "😐",
                'Negative': "😡"
            }
            st.write(f"Sentiment Emoji: {sentiment_emoji[predicted_sentiment]}")

            # Optional detailed output: Preprocessed text and prediction explanation
            cleaned_input = preprocess_text(user_input)
            st.markdown("#### 🔍 Additional Details:")
            st.markdown(f"- **Original Text:** {user_input}")
            st.markdown(f"- **Processed Text:** {cleaned_input}")
            st.markdown(f"- **Prediction:** {predicted_sentiment} {sentiment_emoji[predicted_sentiment]}")

        else:
            # Display warning if input is empty
            st.warning("⚠️ Please enter a valid tweet to analyze.")                
elif menu == "Contact Us":
    st.title(" 📞 Contact Us")
    st.write("Have any questions or feedback? Reach out to us!")
    with st.form("contact_form"):
        email = st.text_input("Your Email")
        query = st.text_area("Your Query")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.success("Thank you for reaching out! We will get back to you shortly.")


