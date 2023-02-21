from flask import Flask, render_template, redirect, request, Response
import pandas as pd
import numpy as np
import re
import fetch_tweets

app = Flask(__name__)


def getKey():
    file = open('keyword.txt', 'r')
    keyword = file.read()
    file.close()
    return keyword


@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        keyword = request.form['key']
        new_search = keyword + " -filter:retweets"
        date_since = request.form['date_since']
        noOfTweet = 100
        fetch_tweets.create_data(new_search, date_since, noOfTweet)
        # df = pd.read_csv('tweets.csv', encoding ='latin1')
        # labels = [row for row in df['TimeStamp']]
        # values = [row for row in df['Score']]
        # print(labels)
        file = open('keyword.txt', 'w')
        file.write(keyword)
        file.close()
        return redirect('/pie')

    return render_template('form.html', keyword=getKey())


@app.route('/line')
def line():
    chart_data = pd.read_csv('tweets.csv', encoding ='latin1')
    chart_data = chart_data.drop(
        columns=['User', 'Tweet', 'Text', 'Location', 'Sentiment', 'Score'], axis=1)
    start_date = pd.to_datetime(chart_data['Date'].head(1).values[0])
    end_date = pd.to_datetime(chart_data['Date'].tail(1).values[0])
    chart_data['Date'] = pd.to_datetime(chart_data['Date'])
    chart_data = chart_data.set_index('Date')
    if start_date == end_date:
        chart_data = chart_data.resample('H').mean().abs()
    else:
        chart_data = chart_data.resample('D').mean().abs()
    labels = [row for row in chart_data.index.astype(str)]
    positive = [row for row in chart_data['Pos_Score']]
    negative = [row for row in chart_data['Neg_Score']]
    neutral = [row for row in chart_data['Neu_Score']]
    return render_template('line.html', labels=labels, positive=positive, negative=negative, neutral=neutral, keyword=getKey())


@app.route('/pie')
def pie():
    data = pd.read_csv('tweets.csv', encoding ='latin1')

    def to_percent(lst):
        temp = len(lst)/len(data)*100
        return temp

    Data = [to_percent(data[data['Sentiment'] == 1]), to_percent(
        data[data['Sentiment'] == -1]), to_percent(data[data['Sentiment'] == 0])]
    labels = ["Positive", "Negative", 'Neutral']
    return render_template('pie.html', labels=labels, values=Data, keyword=getKey())


@app.route('/bar')
def bar():
    chart_data = pd.read_csv('tweets.csv', encoding ='latin1')
    chart_data = chart_data.drop(
        columns=['User', 'Tweet', 'Text', 'Location', 'Sentiment', 'Score'], axis=1)
    start_date = pd.to_datetime(chart_data['Date'].head(1).values[0])
    end_date = pd.to_datetime(chart_data['Date'].tail(1).values[0])
    chart_data['TimeStamp'] = pd.to_datetime(chart_data['TimeStamp'])
    chart_data = chart_data.set_index('TimeStamp')
    if start_date == end_date:
        chart_data = chart_data.resample('H').mean().abs()
    else:
        chart_data = chart_data.resample('D').mean().abs()
    labels = [row for row in chart_data.index.astype(str)]
    positive = [row for row in chart_data['Pos_Score']]
    negative = [row for row in chart_data['Neg_Score']]
    neutral = [row for row in chart_data['Neu_Score']]
    return render_template('bar.html', labels=labels, positive=positive, negative=negative, neutral=neutral, keyword=getKey())


@app.route('/live-chart')
def liveChart():
    file = open('keyword.txt')
    key = file.read()
    file.close()
    df = pd.read_csv('tweets.csv', encoding ='latin1')
    labels = [row for row in df['TimeStamp']]
    values = [row for row in df['Score']]
    # print(labels)
    return render_template('liveChart.html', labels=labels, values=values, keyword=getKey())


@app.route('/live-data')
def liveData():
    file = open('keyword.txt')
    key = file.read()
    file.close()
    keyword = key
    new_search = keyword + " -filter:retweets"
    return fetch_tweets.updater(new_search)


@app.route('/form', methods=['POST', 'GET'])
def get_sentiment():
    if request.method == 'POST':
        word = request.form['input']
        import sentiment_analyzer
        sent = sentiment_analyzer.getSentiment(word)
        negative_score = sent['score']['neg']
        positive_score = sent['score']['pos']
        neutral_score = sent['score']['neu']
        # print(negative_score, positive_score, neutral_score)
        # return sentiment_analyzer.getSentiment(word)
        if(negative_score > positive_score):
            data_sen = "Negative"
        elif(positive_score > neutral_score):
            data_sen = "Positive"
        else:
            data_sen = "Neutral"

    return render_template('form.html', negative_score=negative_score, positive_score=positive_score, neutral_score=neutral_score,
                           data_sen=data_sen, word=word, keyword=getKey())


@app.route('/donut')
def donut():
    data = pd.read_csv('tweets.csv', encoding ='latin1')

    def to_percent(lst):
        temp = len(lst)/len(data)*100
        return temp

    Data = [to_percent(data[data['Sentiment'] == 1]), to_percent(
        data[data['Sentiment'] == -1]), to_percent(data[data['Sentiment'] == 0])]
    labels = ["Positive", "Negative", 'Neutral']
    return render_template('donut.html', labels=labels, values=Data, keyword=getKey())


@app.route('/stacked')
def stacked():
    # this is the same code in bar chart
    chart_data = pd.read_csv('tweets.csv', encoding ='latin1')
    chart_data = chart_data.drop(
        columns=['User', 'Tweet', 'Text', 'Location', 'Sentiment', 'Score'], axis=1)
    start_date = pd.to_datetime(chart_data['Date'].head(1).values[0])
    end_date = pd.to_datetime(chart_data['Date'].tail(1).values[0])
    chart_data['TimeStamp'] = pd.to_datetime(chart_data['TimeStamp'])
    chart_data = chart_data.set_index('TimeStamp')
    if start_date == end_date:
        chart_data = chart_data.resample('H').mean().abs()
    else:
        chart_data = chart_data.resample('D').mean().abs()
    labels = [row for row in chart_data.index.astype(str)]
    positive = [row for row in chart_data['Pos_Score']]
    negative = [row for row in chart_data['Neg_Score']]
    neutral = [row for row in chart_data['Neu_Score']]
    return render_template('stacked.html', labels=labels, positive=positive, negative=negative, neutral=neutral, keyword=getKey())


@app.route('/scatter')
def scatter():
    chart_data = pd.read_csv('tweets.csv', encoding ='latin1')
    chart_data = chart_data.drop(
        columns=['User', 'Tweet', 'Text', 'Location', 'Sentiment'], axis=1)
    start_date = pd.to_datetime(chart_data['Date'].head(1).values[0])
    end_date = pd.to_datetime(chart_data['Date'].tail(1).values[0])
    chart_data['TimeStamp'] = pd.to_datetime(chart_data['TimeStamp'])
    chart_data = chart_data.set_index('TimeStamp')
    # chart_data = chart_data.resample('H').mean()
    chart_data = chart_data.groupby(np.arange(len(chart_data))//5).mean().abs()

    pos = pd.DataFrame(
        {'x': chart_data['Score'], 'y': chart_data['Pos_Score']})
    pos = pos.to_dict('record')
    neg = pd.DataFrame(
        {'x': chart_data['Score'], 'y': chart_data['Neg_Score']})
    neg = neg.to_dict('record')
    neu = pd.DataFrame(
        {'x': chart_data['Score'], 'y': chart_data['Neu_Score']})
    neu = neu.to_dict('record')
    # print(pos)
    return render_template('scatter.html', positive=pos, negative=neg, neutral=neu, keyword=getKey())


@app.route('/cloud/<int:s>')
def cloud(s):
    if s == 2:
        s = -1
    from sklearn.feature_extraction.text import CountVectorizer
    import nltk
    nltk.download('stopwords')
    tw_list = pd.read_csv('tweets.csv', encoding ='latin1')
    # Apply tokenization

    def tokenization(text):
        text = re.split('\W+', text)
        return text

    tw_list['tokenized'] = tw_list['Text'].apply(
        lambda x: tokenization(x.lower()))
    # Removing Stop words
    stopword = nltk.corpus.stopwords.words('english')

    def remove_stopwords(text):
        text = [word for word in text if word not in stopword]
        return text

    tw_list['nonstop'] = tw_list['tokenized'].apply(
        lambda x: remove_stopwords(x))

    # Stemmer
    ps = nltk.PorterStemmer()

    def stemming(text):
        text = [ps.stem(word) for word in text]
        return text

    tw_list['stemmed'] = tw_list['nonstop'].apply(lambda x: stemming(x))

    # join all the words to make a final text field
    tw_list['final'] = tw_list['stemmed'].apply(lambda x: ' '.join(x))
    # def find_hashtags(tweet):
    #     #This function extracts hashtags from the tweets.
    #     return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet)

    # tw_list['hashtags'] = tw_list.Tweet.apply(find_hashtags)
    tw_list = tw_list.drop(columns=['tokenized', 'nonstop', 'stemmed'])

    countVectorizer = CountVectorizer()
    countVector = countVectorizer.fit_transform(
        tw_list['final'][tw_list['Sentiment'] == s])
    print('{} Number of tweets have {} words'.format(
        countVector.shape[0], countVector.shape[1]))
    count_vect_df = pd.DataFrame(
        countVector.toarray(), columns=countVectorizer.get_feature_names())
    # most frequently used words in the tweets
    counts = pd.DataFrame(count_vect_df.sum())
    count_df = counts.sort_values(0, ascending=False).head(30)

    count_df['x'] = count_df.index
    count_df.columns = ['value', 'x']
    data = count_df.to_dict('record')
    return render_template('cloud.html', data=data, keyword=getKey())


@app.route('/table')
def table():
    chart_data = pd.read_csv('tweets.csv', encoding ='latin1')
    chart_data = chart_data.drop(columns=['TimeStamp', 'Date', 'Time', 'Text',
                                 'Location', 'Score', 'Pos_Score', 'Neg_Score', 'Neu_Score'], axis=1)
    chart_data['Sentiment'] = chart_data['Sentiment'].replace(1, 'Positive')
    chart_data['Sentiment'] = chart_data['Sentiment'].replace(-1, 'Negative')
    chart_data['Sentiment'] = chart_data['Sentiment'].replace(0, 'Neutral')
    return render_template('table.html', tables=[chart_data.to_html()], titles=[''], keyword=getKey())


if __name__ == "__main__":
    app.run(debug=True)
