import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')
from collections import defaultdict
# Import necessary libraries for preprocessing
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from contractions import fix

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

# Function to analyze sentiment for specific features
def analyze_sentiment_with_adjectives(row, features):
    sentiment_results = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0, 'adjectives': []})
    for feature in features:
        if feature in row['cleaned_text']:
            # Use the pre-calculated compound score
            compound = row['compound']
            if compound > 0.34:
                sentiment_results[feature]['positive'] += 1
            elif compound < -0.05:
                sentiment_results[feature]['negative'] += 1
            else:
                sentiment_results[feature]['neutral'] += 1
            # Extract adjectives using POS tagging
            adjectives = extract_adjectives(row['cleaned_text'], feature)
            sentiment_results[feature]['adjectives'].extend(adjectives)
    return sentiment_results

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

