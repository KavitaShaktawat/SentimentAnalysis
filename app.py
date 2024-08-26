import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from googleapiclient.discovery import build

# Ensure you have the necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize the YouTube API client
api_key = 'API_KEY'
youtube = build('youtube', 'v3', developerKey='API_KEY')

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Sentiment Analysis Dashboard"

# Text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Function to get YouTube comments from a video URL
def get_youtube_comments(video_url):
    try:
        video_id = video_url.split('v=')[1]
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100  # You can increase this value to get more comments
        )
        response = request.execute()

        comments = []
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
            comments.append(comment)

        return comments
    except Exception as e:
        print(f"Error fetching comments: {str(e)}")
        return []

# Layout of the dashboard
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Sentiment Analysis", className="text-center text-primary mb-4"), width=12)
    ]),

    dbc.Row([
        dbc.Col(html.Div(children='''
            This dashboard allows you to analyze the sentiment of text inputs. 
            Enter a comment or a YouTube video URL to predict the sentiment.
        '''), width=12)
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='sentiment-bar-chart'), width=12)
    ]),

    dbc.Row([
        dbc.Col(html.H2("Enter a YouTube video URL for sentiment analysis:", className="mt-4"), width=12)
    ]),

    dbc.Row([
        dbc.Col(dcc.Input(
            id='input-url',
            placeholder='Enter YouTube video URL here...',
            type='text',
            style={'width': '100%'},
            className="mb-3"
        ), width=12)
    ]),

    dbc.Row([
        dbc.Col(html.Button('Analyze Video Sentiment', id='analyze-video-button', n_clicks=0, className="btn btn-primary"), width=12)
    ]),

    dbc.Row([
        dbc.Col(html.H2("Enter a single comment for sentiment prediction:", className="mt-4"), width=12)
    ]),

    dbc.Row([
        dbc.Col(dcc.Textarea(
            id='input-comment',
            placeholder='Enter your comment here...',
            style={'width': '100%', 'height': 100},
            className="mb-3"
        ), width=12)
    ]),

    dbc.Row([
        dbc.Col(html.Button('Analyze Comment Sentiment', id='analyze-comment-button', n_clicks=0, className="btn btn-primary"), width=12)
    ]),

    dbc.Row([
        dbc.Col(html.Div(id='prediction-result', className="mt-3"), width=12)
    ])
])

# Callback to update the bar chart and predict sentiment for a YouTube video and single comment
@app.callback(
    [Output('sentiment-bar-chart', 'figure'),
     Output('prediction-result', 'children')],
    [Input('analyze-video-button', 'n_clicks'),
     Input('analyze-comment-button', 'n_clicks')],
    [State('input-url', 'value'),
     State('input-comment', 'value')]
)
def analyze_sentiment(video_clicks, comment_clicks, input_url, input_comment):
    ctx = dash.callback_context

    if not ctx.triggered:
        return {}, ''
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'analyze-video-button' and video_clicks > 0 and input_url:
        comments = get_youtube_comments(input_url)
        if not comments:
            return {}, html.Div("No comments found or invalid URL.", style={'color': 'red'})
        
        processed_comments = [preprocess_text(comment) for comment in comments]
        vectorized_comments = vectorizer.transform(processed_comments)
        predictions = model.predict(vectorized_comments)

        df = pd.DataFrame({'Comment': comments, 'Sentiment': predictions})
        sentiment_counts = df['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig = px.bar(sentiment_counts, x='Sentiment', y='Count', title='Sentiment Distribution of Comments',
                     color='Sentiment', color_discrete_map={'Positive':'green', 'Neutral':'blue', 'Negative':'red'})
        
        result = html.Div([
            html.H5("Sentiment analysis of video comments completed."),
            dcc.Graph(figure=fig)
        ])
        return fig, result

    if button_id == 'analyze-comment-button' and comment_clicks > 0 and input_comment:
        processed_comment = preprocess_text(input_comment)
        vectorized_comment = vectorizer.transform([processed_comment])
        prediction = model.predict(vectorized_comment)[0]
        color = 'green' if prediction == 'Positive' else 'red' if prediction == 'Negative' else 'blue'
        result = html.Div([
            html.H5(f"The sentiment of the entered comment is: {prediction}", style={'color': color}),
            html.P(f"Original comment: {input_comment}")
        ])

        # Update bar chart (for demonstration purposes, can be customized further)
        sentiment_counts = pd.DataFrame({'Sentiment': ['Positive', 'Neutral', 'Negative'], 'Count': [0, 0, 0]})
        if prediction == 'Positive':
            sentiment_counts.loc[0, 'Count'] = 1
        elif prediction == 'Neutral':
            sentiment_counts.loc[1, 'Count'] = 1
        elif prediction == 'Negative':
            sentiment_counts.loc[2, 'Count'] = 1
        
        fig = px.bar(sentiment_counts, x='Sentiment', y='Count', title='Sentiment Analysis of Entered Comment',
                     color='Sentiment', color_discrete_map={'Positive':'green', 'Neutral':'blue', 'Negative':'red'})
        
        return fig, result

    return {}, ''

if __name__ == '__main__':
    app.run_server(debug=True)
