import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib

# load the model from disk
cnb = joblib.load('finalized_model.sav')
cv = joblib.load('vectorizer.sav')


def getSentiment(review):
  review_vector = cv.transform([review]) # vectorizing
  prediction = cnb.predict(review_vector)
  score = SentimentIntensityAnalyzer().polarity_scores(review)
#   print(score)

  if prediction ==[1] or score['compound'] >= 0.05 :
    sentiment = 1
    prob = cnb.predict_proba(review_vector).max()

  if prediction ==[-1] and score['compound'] <= - 0.05:
    sentiment = -1
    prob = cnb.predict_proba(review_vector).max()
  if score['compound'] <= - 0.05:
    sentiment = -1
    prob = cnb.predict_proba(review_vector).max()

  if prediction == [0] or (score['compound'] > - 0.05 and score['compound'] < 0.05):
    sentiment = 0
    prob = 0
  

  if(sentiment == 1):
      positive_score = score['compound']
      negative_score = 0
      neutral_score = 0
  if(sentiment == -1):
      positive_score = 0
      negative_score = score['compound']
      neutral_score = 0
  if(sentiment == 0):
      negative_score = 0
      positive_score = 0
      neutral_score = score['compound']

  return {'sentiment': sentiment, 'score': score, 'pos_score': positive_score, 'neg_score': negative_score, 'neu_score': neutral_score}

# review = input("enter text: ")
# print(getSentiment(review)['sentiment'])