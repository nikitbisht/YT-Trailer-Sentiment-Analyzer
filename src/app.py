import streamlit as st
import pandas as pd
import requests
import re
import pickle as pk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

def get_all_comment(video_id,api_key,max_comments):
    comments_data = []
    next_page_token = None
    max_comment_reached = False
    max_count=0
    while not max_comment_reached:
        url = f"https://www.googleapis.com/youtube/v3/commentThreads"
        params = {
            'part' : 'snippet',
            'maxResult':100,
            'videoId':video_id,
            'key': api_key,
            'pageToken':next_page_token
        }
        response = requests.get(url,params=params)
        data = response.json()

        # Extract comments from response
        for item in data['items']:
            comment_info = item['snippet']['topLevelComment']['snippet']
            comment = comment_info['textOriginal']
            published_at = comment_info['publishedAt']
            author = comment_info['authorDisplayName']
            max_count+=1
            comments_data.append({
                'Author': author,
                'PublishedAt': published_at,
                'Comment': comment
            })
            
            #Check max comment reach or not
            if max_count >= max_comments:
                max_comment_reached = True
                return comments_data
        
        # Check if there are more comments available
        if 'nextPageToken' in data and not max_comment_reached:
            next_page_token = data['nextPageToken']
        else:
            break  # No more comments available
    return comments_data



#Data cleaning
def transform_text(text):
    wt = WordNetLemmatizer()
    corpus = []
    patten = re.compile('<.*?>')
    text = patten.sub(r' ',text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [wt.lemmatize(word) for word in text if word not in stopwords.words('english')]
    text = ' '.join(text)
    return text


#Extract Video Id
def extract_video_id(url):
    if 'youtu.be/' in url:
        return url.split('youtu.be/')[-1].split('?')[0]
    elif 'youtube.com/watch?v=' in url:
        return url.split('watch?v=')[-1].split('&')[0]
    elif 'youtube.com/embed/' in url:
        return url.split('embed/')[-1].split('?')[0]
    else:
        return None

#fetch Video details
def get_video_details(api_key, video_id):
    url = f'https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'items' in data and len(data['items']) > 0:
            video_details = data['items'][0]
            title = video_details['snippet']['title']
            like_count = video_details['statistics'].get('likeCount', 'Not available')
            views_count = video_details['statistics'].get('viewCount','not available')
            comment_count = video_details['statistics'].get('commentCount', 'not available')
            # Dislike count is no longer available in API
            return title, like_count , views_count , comment_count
    return -1,-1,-1,-1


st.write("""
    <style>
    .big_font {
        font-size:32px ;
        padding: 0px 10px;
        border-radius: 05px;
        border: 1px solid white;
    }
    </style>
    """, unsafe_allow_html=True)


api_key = 'AIzaSyB1t7szBl0tqD7z3ClXYiz54EqBx7vwzfM'

st.title("Yt Trailer Sentimental Analyzer")
input_url = st.text_input("Enter The Url")
max_comments = 500

model=pk.load(open('models/model.pkl','rb'))
cv = pk.load(open('models/vectorizer.pkl','rb'))

if st.button("Predict"):
    if not input_url:
        st.error("Enter the Url")
        exit()

    video_id = extract_video_id(input_url)
    title, like_count , views_count, comment_count = get_video_details(api_key, video_id)
    if like_count == -1:
        st.error("Wrong Video Id")
        exit()
    data = get_all_comment(video_id,api_key,max_comments)
    
    st.header(title)
    st.write(f'<p class = "big_font" style = "background-color: green;">Views: {views_count}</p>',unsafe_allow_html = True)
    st.write(f'<p class = "big_font" style = "background-color: orange;">Likes: {like_count}</p>',unsafe_allow_html = True)
    st.write(f'<p class = "big_font" style = "background-color: red;">Comment: {comment_count}</p>',unsafe_allow_html = True)

    st.write(f'<p class = "big_font" style = "text-align:center; font-size: 24px;">Prediction of Positive and Negative Comments</p>',unsafe_allow_html = True)
    df = pd.DataFrame(data)
    df = df.drop_duplicates().copy()
    df['transform_text'] = df['Comment'].apply(transform_text)
    y = cv.transform(df['transform_text']).toarray()
    x = model.predict(y)
    predictions = np.array(x)

    # Count the number of negative and positive reviews
    num_negative_reviews = np.sum(predictions == 0)
    num_positive_reviews = np.sum(predictions == 1)

    # Plotting
    labels = ['Negative', 'Positive']
    sizes = [num_negative_reviews, num_positive_reviews]
    colors = ['red', 'green']

    plt.bar(labels, sizes, color=colors)
    plt.title('Distribution of Comments')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Comments')
    plt.savefig('images/plot.png')
    st.image('images/plot.png')
