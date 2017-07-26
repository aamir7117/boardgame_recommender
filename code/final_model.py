import pandas as pd
import numpy as np
from pymongo import MongoClient
import graphlab
import random
import gc

client = MongoClient()
db = client['bgg_data']
games = db['games']
all_info = db['all_info']
master = db['master']

def make_pandas_df(mongodb_collection):
    '''
    Makes a pandas df from a mongodb collection of ratings
    df = make_pandas_df(master)
    '''
    cols = ['game_id','game_name','player_name','player_rating']
    df = pd.DataFrame(list(mongodb_collection.find()),columns=cols)
    return df

def split_data(df,num_ratings):
    '''
    df_train, df_test = split_data(df,num_ratings)
    df, holdout = split_data(df,num_ratings)
    '''
    counts = df['player_name'].value_counts()
    active_users = counts[counts>1]  #We will only take ratings from active users
    holdout = df[df['player_name'].isin(active_users.index)].sample(num_ratings)
    return df.drop(holdout.index), holdout

def make_cosine_similarity_model(sf_train,sf_test,verbose=True):
    '''
    Make a cosine similarity model from SFrames, sf_train and sf_test
    print the average_rmse, Precision and recall metrics
    '''
    if verbose==True:
        print "making cosine similarity model"
    cos_sim_model = graphlab.recommender.item_similarity_recommender.create(sf_train, user_id='player_name', item_id='game_id',target='player_rating',similarity_type='cosine')
    if verbose==True:
        print "evaluating model"
    eval_cos_sim = cos_sim_model.evaluate_rmse(sf_test,target='player_rating')
    if verbose==True:
        print "average_rmse_per_user = {}".format(np.average(eval_cos_sim['rmse_by_user']['rmse']))
        print "average_rmse_per_item = {}".format(np.average(eval_cos_sim['rmse_by_item']['rmse']))
        print "average_rmse_overall = {}".format(np.average(eval_cos_sim['rmse_overall']))

    prec_rec_cos_sim = cos_sim_model.evaluate(sf_test,metric='precision_recall',cutoffs=[10,50,100,500])
    if verbose==True:
        print prec_rec_cos_sim['precision_recall_overall']
    return cos_sim_model




if __name__ == '__main__':
    df = make_pandas_df(master)
    print "splitting data into holdout and df"
    df, holdout = split_data(df,20)
    print "splitting data into df_train an df_test"
    df_train, df_test = split_data(df, 10)
    print "making sf_train (turi SFrame)"
    sf_train = graphlab.SFrame(df_train)
    print "making sf_test (turi SFrame)"
    sf_test = graphlab.SFrame(df_test)
    print "making sf_holdout (turi SFrame)"
    sf_holdout = graphlab.SFrame(holdout)
    print "shuffling the SFrame, sf_train"
    sf_train = graphlab.cross_validation.shuffle(sf_train)
    gc.collect()
