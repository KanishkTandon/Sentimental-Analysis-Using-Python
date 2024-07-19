import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load the data
df = pd.read_csv(r'C:\Users\xpert\Downloads\archive\Reviews.csv')
print(df.shape)
df = df.head(500)
print(df.shape)

# Create a SentimentIntensityAnalyzer object
sia = SentimentIntensityAnalyzer()

# Define a function to classify sentiment as positive, negative, or neutral
def classify_sentiment(text):
    scores = sia.polarity_scores(text)
    compound_score = scores['compound']
    if compound_score > 0.5:
        return 'positive'
    elif compound_score < -0.5:
        return 'negative'
    else:
        return 'neutral'

# Apply the classify_sentiment function to each review
df['Sentiment'] = df['Text'].apply(classify_sentiment)

# Print the distribution of sentiment labels
print(df['Sentiment'].value_counts())

# Plot the distribution of sentiment labels
ax = df['Sentiment'].value_counts().plot(kind='bar', title='Sentiment Distribution')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Count')
plt.show()

# Load the Roberta model and tokenizer
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Define a function to classify sentiment using the Roberta model
def classify_sentiment_roberta(text):
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    if scores[2] > scores[0] and scores[2] > scores[1]:
        return 'positive'
    elif scores[0] > scores[1] and scores[0] > scores[2]:
        return 'negative'
    else:
        return 'neutral'

# Apply the classify_sentiment_roberta function to each review
df['Sentiment_Roberta'] = df['Text'].apply(classify_sentiment_roberta)

# Print the distribution of sentiment labels using the Roberta model
print(df['Sentiment_Roberta'].value_counts())

# Plot the distribution of sentiment labels using the Roberta model
ax = df['Sentiment_Roberta'].value_counts().plot(kind='bar', title='Sentiment Distribution (Roberta)')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Count')
plt.show()