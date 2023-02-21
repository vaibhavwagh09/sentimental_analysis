import tweepy
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import string
import nltk
import time
from csv import writer, DictWriter
from sentiment_analyzer import getSentiment

def cleanUpTweet(txt):
    txt = re.sub(r'https?:\/\/(www\.)?[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)', '', txt, flags=re.MULTILINE) # to remove links that start with HTTP/HTTPS in the tweet
    txt = re.sub(r'[-a-zA-Z0–9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0–9@:%_\+.~#?&//=]*)', '', txt, flags=re.MULTILINE) # to remove other url links
    # hashes = re.findall(r'\B#\w*[a-zA-Z]+\w*', txt)
    txt  = "".join([char for char in txt if char not in string.punctuation])
    txt = re.sub(r"#(\w+)", ' ', txt)
    txt = re.sub(r"@(\w+)", ' ', txt)
    txt = re.sub(r"\d", "", txt)
    txt = txt.lower()
    txt = re.sub(r'RT : ', '', txt)   
    return txt

# Authentication credentials
access_token = "1484086414828781568-r9Dqa9YJ1BKlYcINGR0jQxRIxHPb0F"
access_token_secret = "erGha0z3VuBPg1o08ShkLQWMUIkZtOY38dx42ur9W8krw"
consumer_key = "zLwbZVPlIOzda1UYjgTRBIEJ7"
consumer_secret = "BRcEdMDJytEvu2hXLoJqSycGNptYe3qxiIThSdCk6cbIufSEaT"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
def updater(new_search):
  
  tweets = tweepy.Cursor(api.search_tweets, q=new_search, lang='en' ).items(1)
  info =[]
  for tweet in tweets:
    date_string = str(tweet.created_at)
    DateTime =  pd.to_datetime(date_string, infer_datetime_format=True)
    prediction = getSentiment(cleanUpTweet(tweet.text))
    # print(DateTime)
    info = {"TimeStamp":DateTime,
    "Date": str(DateTime.date()),
    "Time": str(DateTime.time()),
    "User": tweet.user.screen_name,
    "Tweet": tweet.text,
    "Text": cleanUpTweet(tweet.text),
    "Location": tweet.user.location,
    "Sentiment": prediction['sentiment'],
    "Score": prediction['score']['compound'],
    "Pos_Score": prediction['pos_score'],
    "Neg_Score": prediction['neg_score'],
    "Neu_Score": prediction['neu_score']
    }

  
  tw = pd.read_csv('tweets.csv', encoding ='latin1')
  if str(tw['TimeStamp'].tail(1).values[0]) != str(DateTime):
    # print('Hiiiiiiiiiii-----------------------------')
    fields = ['TimeStamp','Date','Time','User','Tweet','Text','Location','Sentiment','Score','Pos_Score','Neg_Score','Neu_Score']
    
    
    with open('tweets.csv', 'a') as csv_file:
      csv_writter = DictWriter(csv_file, fieldnames = fields)
      try:
        csv_writter.writerow(info)
        return info
      except UnicodeEncodeError:
        return {'s': 0}
      finally:
        csv_file.close()
  else:
    return "0"
    time.sleep(5)
  
  time.sleep(10)


def create_data( new_search, date_since, noOfTweet):
  date_since =  pd.to_datetime(date_since, infer_datetime_format=True)
  date_since =  date_since.date() + timedelta(days=1)
  column = [ 'TimeStamp', "Date", "Time", "User", "Tweet", "Text",'Location', 'Sentiment', 'Score', 'Pos_Score', 'Neg_Score', 'Neu_Score']
  data =[]
  while datetime.today().date()+timedelta(days=2) != date_since:
    try:
      tweets = tweepy.Cursor(api.search_tweets, q=new_search, lang='en', until=date_since).items(noOfTweet)
      for tweet in tweets:
        # print(date_string)
        date_string = str(tweet.created_at)
        DateTime =  pd.to_datetime(date_string, infer_datetime_format=True)
        prediction = getSentiment(cleanUpTweet(tweet.text))
        data.append([DateTime, str(DateTime.date()), str(DateTime.time()), tweet.user.screen_name, tweet.text, cleanUpTweet(tweet.text), tweet.user.location, prediction['sentiment'], prediction['score']['compound'], prediction['pos_score'], prediction['neg_score'], prediction['neu_score']])

      # updater()
    except tweepy.errors.TooManyRequests:
      print("request limit reched try after 5 min.")

    print(date_since)
    date_since = date_since + timedelta(days=1)
  
  tw_list = pd.DataFrame(data, columns=column)
  tw_list = tw_list.set_index('TimeStamp')

  tw_list.to_csv('tweets.csv')


# keyword = "would putin"
# new_search = keyword + " -filter:retweets"
# date_since = "2022-03-24"
# noOfTweet = 100
# create_data(new_search, date_since, noOfTweet)