import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from googleapiclient.discovery import build
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Ensure you have the necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Your YouTube API key
API_KEY = "API_KEY"
VIDEO_ID = "ISc5_x_3MWM"

# YouTube API client
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Initialize list to store comments data
comments_data = []

# Cleaning functions
def clean_text(text):
    return text.encode('utf-8', 'ignore').decode('utf-8')

def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

def scrape_comments_with_replies():
    response = youtube.commentThreads().list(
        part='snippet',
        videoId=VIDEO_ID,
        maxResults=100,
        textFormat="plainText"
    ).execute()

    while True:
        for item in response["items"]:
            comment = item["snippet"]['topLevelComment']["snippet"]
            name = clean_text(comment["authorDisplayName"])
            comment_text = clean_text(comment["textDisplay"])
            published_at = comment['publishedAt']
            likes = comment['likeCount']
            replies = item["snippet"]['totalReplyCount']
            sentiment = get_sentiment(comment_text)
            comments_data.append([name, comment_text, published_at, likes, replies, sentiment])

            if replies > 0:
                parent_id = item["snippet"]['topLevelComment']["id"]
                response2 = youtube.comments().list(
                    part='snippet',
                    maxResults=100,
                    parentId=parent_id,
                    textFormat="plainText"
                ).execute()

                for item2 in response2["items"]:
                    reply_comment = item2["snippet"]
                    name = clean_text(reply_comment["authorDisplayName"])
                    comment_text = clean_text(reply_comment["textDisplay"])
                    published_at = reply_comment['publishedAt']
                    likes = reply_comment['likeCount']
                    replies = ''
                    sentiment = get_sentiment(comment_text)
                    comments_data.append([name, comment_text, published_at, likes, replies, sentiment])

        if 'nextPageToken' in response:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=VIDEO_ID,
                pageToken=response["nextPageToken"],
                maxResults=100,
                textFormat="plainText"
            ).execute()
        else:
            break

    df = pd.DataFrame(comments_data, columns=['Name', 'Comment', 'Time', 'Likes', 'Reply Count', 'Sentiment'])
    df['Comment'] = df['Comment'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    df = df[df['Comment'] != '']
    return df

df = scrape_comments_with_replies()

# Text preprocessing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def remove_stopwords(text):
    return " ".join([word for word in text.split() if word.lower() not in stop_words])

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def clean_special_chars(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def to_lowercase(text):
    return text.lower()

def preprocess_text(text):
    text = to_lowercase(text)
    text = clean_special_chars(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

df['Processed_Comment'] = df['Comment'].apply(preprocess_text)

# Vectorize the comments
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['Processed_Comment'])
y = df['Sentiment']

# Handle imbalanced data
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Save the model and vectorizer
import joblib
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Example predictions
new_comments = ["I love this video!", "This is the worst video I've ever seen."]
new_X = vectorizer.transform([preprocess_text(comment) for comment in new_comments])
predictions = model.predict(new_X)
print(predictions)

# Data visualization
plt.figure(figsize=(10, 6))
sns.countplot(x='Sentiment', data=df, order=['Positive', 'Neutral', 'Negative'])
plt.title('Sentiment Analysis of Comments')
plt.xlabel('Sentiment')
plt.ylabel('Number of Comments')
plt.tight_layout()
plt.savefig('sentiment_analysis.png')
plt.show()
