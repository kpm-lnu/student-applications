import tweepy
import json
from datetime import datetime, timedelta


consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)


hashtag = '#neonazi'
start_date = datetime(2022, 2, 24)
end_date = datetime(2023, 1, 30)


def search_tweets_by_date(date):
    formatted_date = date.strftime('%Y-%m-%d')
    query = f'{hashtag} since:{formatted_date} until:{formatted_date}'
    tweets = api.search_tweets(q=query, count=100)
    return len(tweets)


data = {}


current_date = start_date
while current_date <= end_date:
    tweet_count = search_tweets_by_date(current_date)
    data[current_date.strftime("%Y-%m-%d")] = tweet_count
    current_date += timedelta(days=1)


filename = f"{hashtag[1:]}-word.json" 
with open(filename, 'w') as f:
    json.dump(data, f)

print(f"Data saved to {filename}")