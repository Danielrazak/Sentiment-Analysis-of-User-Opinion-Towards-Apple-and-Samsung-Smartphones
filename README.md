# Sentiment-Analysis-of-User-Opinion-Towards-Apple-and-Samsung-Smartphones
# Sentiment Analysis Dashboard

This project is a Streamlit-based web application that performs sentiment analysis on social media data to determine public opinion towards Apple and Samsung smartphones. It leverages Natural Language Processing (NLP) and machine learning techniques to analyze tweets and provide insights into user sentiments across different smartphone features like design, performance, and battery life.

## Features

- **Sentiment Analysis**: Classifies sentiments of tweets as positive, neutral, or negative using a pre-trained SVM model.
- **Feature-Based Insights**: Breaks down sentiments by smartphone features to help understand public perception in detail.
- **Interactive Dashboard**: Provides an interactive UI for users to explore sentiment trends, compare smartphone models, and analyze real-time data.
- **Data-Driven Decision Making**: Aids smartphone companies in understanding consumer preferences and improving their products based on real user feedback.

## Built With

- **[Streamlit](https://streamlit.io/)** - The framework for building the web app.
- **[Pandas](https://pandas.pydata.org/)** - For data manipulation and analysis.
- **[Plotly](https://plotly.com/)** - For creating interactive charts.
- **[NLTK](https://www.nltk.org/)** - For text preprocessing and sentiment analysis.
- **[Scikit-learn](https://scikit-learn.org/stable/)** - For machine learning model implementation.

## Getting Started

### Prerequisites

Before running this project, you will need to have Python installed on your system. Python 3.8 or later is recommended. You will also need pip to install the following Python libraries.

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/sentiment-analysis-app.git
cd sentiment-analysis-app
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### Usage

Run the Streamlit application:

```bash
streamlit run smartaapp.py
```

Navigate to `http://localhost:8501` in your web browser to view the app.

## How It Works

1. **Data Collection**: The app pulls data from a pre-defined dataset of tweets that mention Apple and Samsung smartphones.
2. **Preprocessing**: Tweets are cleaned and preprocessed to extract meaningful text for analysis.
3. **Sentiment Analysis**: The app uses a trained SVM model to predict sentiment from text data.
4. **Visualization**: Results are visualized in an interactive dashboard that allows users to filter data by smartphone model and features.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Daniel Razak - http://www.linkedin.com/in/danielrzak

Project Link: https://github.com/Danielrazak/Sentiment-Analysis-of-User-Opinion-Towards-Apple-and-Samsung-Smartphones
