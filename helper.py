from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


extract = URLExtract()

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch number of media messages
    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    # fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x,df

def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    #temp = df[df['user'] != 'group_notification']
    temp = df[df['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp.loc[:, 'message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):

    f = open('stop_hinglish.txt','r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(), columns=['Emoji', 'Count'])
    return emoji_df

def monthly_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby('only_date').count()['message'].reset_index()

    return timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap


# Sentiment functions

def sentiment_data(selected_user,df):
    sentiments = SentimentIntensityAnalyzer()
    df["positive"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]]
    df["negative"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]]
    df["neutral"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]]

    return selected_user,df

def score(a, b, c):
    if (a > b) and (a > c):
        print("Positive ")
    if (b > a) and (b > c):
        print("Negative")
    if (c > a) and (c > b):
        print("Neutral")


    #bar graph of sentiment score

def sentiment_score(selected_user,df):
    sentiment_labels = ['Positive', 'Negative', 'Neutral']
    x = sum(df["positive"])
    y = sum(df["negative"])
    z = sum(df["neutral"])
    sentiment_scores = [x, y, z]
    return sentiment_labels, sentiment_scores




    #pie chart
def sentiment_score_percent(selected_user, df):
    sentiment_labels = ['Positive', 'Negative', 'Neutral']
    x = sum(df["positive"])
    y = sum(df["negative"])
    z = sum(df["neutral"])
    sentiment_counts = [x, y, z]
    return sentiment_labels, sentiment_counts

    #sentiment distribution
def sentiment_distribution():
    pass




    # word cloud
def sentiment_wordcloud(selected_user, df):


    # Filter messages based on sentiment
    positive_messages = ' '.join(df[df['positive'] > df['negative']]['message'])
    negative_messages = ' '.join(df[df['negative'] > df['positive']]['message'])

    # Generate word clouds
    positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_messages)
    negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_messages)
    return positive_wordcloud, negative_wordcloud


    #sentiment correlation
def sentiment_relation(selected_user, df):
    sentiment_df = df[['positive', 'negative', 'neutral']]
    correlation_matrix = sentiment_df.corr()
    return correlation_matrix

    # sentiment trend over time
def sentiment_trend_over_time(selected_user, df):
    start_date = pd.to_datetime('2022-09-01')
    end_date = pd.to_datetime('2023-01-31')

    # Filter for rows between the start and end date
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    filtered_df.set_index('date', inplace=True)

    # Resample to get daily sentiment averages
    sentiment_daily = filtered_df.resample('D').agg({'positive': 'mean', 'negative': 'mean', 'neutral': 'mean'})
    return sentiment_daily












