import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from apiclient.discovery import build
from textblob import TextBlob  # Import TextBlob for sentiment analysis
import csv
import codecs

# Replace with your own API key
API_KEY = "API_KEY"

# Replace with the YouTube video ID you want to scrape comments from
VIDEO_ID = "ISc5_x_3MWM"

# Create a YouTube API client
youtube = build('youtube', 'v3', developerKey=API_KEY)

# Initialize a list to store the comments data
comments_data = []

def clean_text(text):
    # Clean up text (example: remove non-UTF-8 characters)
    return text.encode('utf-8', 'ignore').decode('utf-8')

def get_sentiment(text):
    analysis = TextBlob(text)
    # Classify sentiment
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

def scrape_comments_with_replies():
    # Get the initial comment threads
    response = youtube.commentThreads().list(
        part='snippet',
        videoId=VIDEO_ID,
        maxResults=100,
        textFormat="plainText"
    ).execute()

    # Process the comment threads
    while True:
        for item in response["items"]:
            # Extract the top-level comment data
            comment = item["snippet"]['topLevelComment']["snippet"]
            name = clean_text(comment["authorDisplayName"])
            comment_text = clean_text(comment["textDisplay"])
            published_at = comment['publishedAt']
            likes = comment['likeCount']
            replies = item["snippet"]['totalReplyCount']
            
            # Perform sentiment analysis
            sentiment = get_sentiment(comment_text)

            # Add the top-level comment data to the list
            comments_data.append([name, comment_text, published_at, likes, replies, sentiment])

            # If the comment has replies, get them
            if replies > 0:
                parent_id = item["snippet"]['topLevelComment']["id"]
                response2 = youtube.comments().list(
                    part='snippet',
                    maxResults=100,
                    parentId=parent_id,
                    textFormat="plainText"
                ).execute()

                for item2 in response2["items"]:
                    # Extract the reply comment data
                    reply_comment = item2["snippet"]
                    name = clean_text(reply_comment["authorDisplayName"])
                    comment_text = clean_text(reply_comment["textDisplay"])
                    published_at = reply_comment['publishedAt']
                    likes = reply_comment['likeCount']
                    replies = ''
                    
                    # Perform sentiment analysis
                    sentiment = get_sentiment(comment_text)

                    # Add the reply comment data to the list
                    comments_data.append([name, comment_text, published_at, likes, replies, sentiment])

        # If there's a next page, get it
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

    # Convert the comments data to a Pandas DataFrame
    df = pd.DataFrame(comments_data, columns=['Name', 'Comment', 'Time', 'Likes', 'Reply Count', 'Sentiment'])

    # Data cleaning: handle empty comments
    df['Comment'] = df['Comment'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    df = df[df['Comment'] != '']  # Remove rows with empty comments

    # Data visualization
    # Example: Plotting number of comments per sentiment category
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Sentiment', data=df, order=['Positive', 'Neutral', 'Negative'])
    plt.title('Sentiment Analysis of Comments')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Comments')
    plt.tight_layout()
    plt.savefig('sentiment_analysis.png')  # Save the plot as an image
    plt.show()

    return "Successful! Check the cleaned CSV file and visualization."

# Run the function to scrape comments, perform sentiment analysis, and visualize
scrape_comments_with_replies()
