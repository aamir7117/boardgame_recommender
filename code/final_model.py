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

def make_cosine_similarity_model(sf_train,sf_test):
    '''
    Returns a cosine similarity model using graphlab's item_similarity_recommender
    Prints the average_rmse, Precision and recall metrics
    '''
    print "making cosine similarity model"
    cos_sim_model = graphlab.recommender.item_similarity_recommender.create(sf_train, user_id='player_name', item_id='game_id',target='player_rating',similarity_type='cosine')
    print "evaluating model"
    eval_cos_sim = cos_sim_model.evaluate_rmse(sf_test,target='player_rating')
    print "average_rmse_per_user = {}".format(np.average(eval_cos_sim['rmse_by_user']['rmse']))
    print "average_rmse_per_item = {}".format(np.average(eval_cos_sim['rmse_by_item']['rmse']))
    print "average_rmse_overall = {}".format(np.average(eval_cos_sim['rmse_overall']))

    prec_rec_cos_sim = cos_sim_model.evaluate(sf_test,metric='precision_recall',cutoffs=[10,50,100,500])
    print prec_rec_cos_sim['precision_recall_overall']
    return cos_sim_model

def make_jaccard_similarity_model(sf_train,sf_test):
    '''
    Returns a jaccard similarity model using graphlab's item_similarity_recommender
    Prints the average_rmse, Precision and recall metrics
    '''
    print "making jaccard similarity model"
    jac_sim_model = graphlab.recommender.item_similarity_recommender.create(sf_train, user_id='player_name', item_id='game_id',target='player_rating',similarity_type='jaccard')
    print "evaluating model"
    eval_jac_sim = jac_sim_model.evaluate_rmse(sf_test,target='player_rating')
    print "average_rmse_per_user = {}".format(np.average(eval_jac_sim['rmse_by_user']['rmse']))
    print "average_rmse_per_item = {}".format(np.average(eval_jac_sim['rmse_by_item']['rmse']))
    print "average_rmse_overall = {}".format(np.average(eval_jac_sim['rmse_overall']))

    jac_sim_model = graphlab.recommender.item_similarity_recommender.create(sf_train_stripped, user_id='player_name', item_id='game_id',similarity_type='jaccard')
    prec_rec_jac_sim = jac_sim_model.evaluate(sf_test_stripped,metric='precision_recall',cutoffs=[10,50,100,500])
    print prec_rec_jac_sim['precision_recall_overall']
    return jac_sim_model

def make_pearson_similarity_model():
    '''
    Returns a pearson_sim_model similarity model using graphlab's item_similarity_recommender
    Prints the average_rmse, Precision and recall metrics
    '''
    print "making pearson similarity model"
    pearson_sim_model = graphlab.recommender.item_similarity_recommender.create(sf_train, user_id='player_name', item_id='game_id',target='player_rating',similarity_type='pearson')
    print "evaluating model"
    eval_pearson_sim = pearson_sim_model.evaluate_rmse(sf_test,target='player_rating')
    print "average_rmse_per_user = {}".format(np.average(eval_pearson_sim['rmse_by_user']['rmse']))
    print "average_rmse_per_item = {}".format(np.average(eval_pearson_sim['rmse_by_item']['rmse']))
    print "average_rmse_overall = {}".format(np.average(eval_pearson_sim['rmse_overall']))
    prec_rec_pearson = pearson_sim_model.evaluate(sf_test_stripped,metric='precision_recall',cutoffs=[10,50,100,500])
    print prec_rec_pearson['precision_recall_overall']
    return pearson_sim_model

def make_random_recommendations(cutoffs):
    '''
    for a list of cutoffs like [10,50,100], make random recommendations and check the precision of those recommendations
    '''
    possible_counts = df['player_name'].value_counts()
    number_games_rated = random.choice(possible_counts)
    games_rated = random.sample(df['game_id'],number_games_rated)
    precisions = defaultdict(list)
    for cutoff in cutoffs:
        rand_recommendations = random.sample(df['game_id'],cutoff)
        precision = (len(set(rand_recommendations) & set(games_rated)) * 1.) / cutoff
        precisions[cutoff] += [precision]
    return precisions


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
