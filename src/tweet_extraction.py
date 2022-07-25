import pandas as pd
import numpy as np
import requests
import tweepy
import config
from tqdm import tqdm
import re

def get_user_tweets(api, twitter_username):
    # extract
    response = api.user_timeline(screen_name=twitter_username, count=200, exclude_replies=True, include_rts=False,  tweet_mode = "extended")

    cols = [x for x in dir(response[0]) if '__' not in x]
    cols.remove('_api')
    cols.remove('_json')

    # prep for df
    tweets_obj_list = []
    for i in range(len(response)):
        row_obj = dict()
        for col in cols:
            try:
                row_obj[col] = response[i]._json[col]
            except:
                row_obj[col] = None
        tweets_obj_list.append(row_obj)

    df_tweets = pd.DataFrame(tweets_obj_list)

    return df_tweets

def extract_tweets():
    auth = tweepy.OAuth1UserHandler(config.API_KEY, config.API_SECRET_KEY, config.ACCESS_TOKEN, config.SECRET_TOKEN)
    api = tweepy.API(auth)

    twitter_usernames = ['wojespn', 'stephenasmith', 'maxkellerman', 'thesteinline', 'RicBucher']

    # extract and convert to csv
    dfs_list = []
    for user in tqdm(twitter_usernames):
        df_user_tweets = get_user_tweets(api, user)
        dfs_list.append(df_user_tweets)

    df_agg_tweets = pd.concat(dfs_list, axis=0)
    df_agg_tweets.to_csv('../data/tweets_raw.csv')

    # data preprocessing
    tweets_list = list(df_agg_tweets['full_text'])
    
    # remove links
    tweets_list = [re.sub(r'http\S+', '', x).replace(':', '').replace('\n', '') for x in tweets_list]

    # remove emojis
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)

    tweets_list = [emoji_pattern.sub(r'', x) for x in tweets_list]

    # write out in a txt file
    file_name = '../data/processed_tweets.txt'
    with open(file_name, 'w') as fp:
        for tweet in tweets_list:
            if tweet not in ['', ' ']:
                fp.write(tweet + '\n\n')
    fp.close()

if __name__ == "__main__":
    extract_tweets()