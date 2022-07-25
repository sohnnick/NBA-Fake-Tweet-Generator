import pandas as pd
import numpy as np
from tqdm import tqdm
import gpt_2_simple as gpt2
import os
import re
import tensorflow as tf
import sys

# global vars
DATA_PATH = '../data/'
TRAIN_DATA_PATH = DATA_PATH + 'processed_tweets.txt'
MODEL_PATH = '../models/'
CHECKPOINT_PATH = MODEL_PATH + 'checkpoint/'

TOPIC = 'nba_v07252022'
MODEL_NAME = "124M"

def train_gpt2_model():
    # if no model folder -> create folder
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)

    # if model is not downloaded -> download pretrained GPT model
    if not os.path.exists(MODEL_PATH+MODEL_NAME):
        gpt2.download_gpt2(model_name=MODEL_NAME, model_dir=MODEL_PATH)

    # if topic does not exist in checkpoing -> finetune GPT model
    if not os.path.exists(CHECKPOINT_PATH+TOPIC):
        sess = gpt2.start_tf_sess()

        gpt2.finetune(
            sess,
            dataset=TRAIN_DATA_PATH,
            model_name=MODEL_NAME,
            run_name=TOPIC,
            checkpoint_dir=CHECKPOINT_PATH,
            steps=50,
            restore_from="fresh",
            print_every=1,
            reuse=tf.compat.v1.AUTO_REUSE
        )

        gpt2.reset_session(sess)

def generate_tweets(num_samples=1):
    # run the model training -> if all the models + checkpoint exist -> it will skip
    train_gpt2_model()

    # load GPT model + checkpoint
    # tf.compat.v1.reset_default_graph()
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(
        sess,
        checkpoint='latest',
        run_name=TOPIC,
        checkpoint_dir=CHECKPOINT_PATH,
        reuse=tf.compat.v1.AUTO_REUSE
    )

    # generate tweet(s) and save as file in data folder
    gpt2.generate_to_file(
        sess,
        length=50,
        nsamples=num_samples,
        run_name=TOPIC,
        checkpoint_dir=CHECKPOINT_PATH,
        destination_path=DATA_PATH+'generated_tweets.txt',
    )

    gpt2.reset_session(sess)

if __name__ == "__main__":
    try:
        num_samples = int(sys.argv[1])
        generate_tweets(num_samples)

    except:
        generate_tweets()