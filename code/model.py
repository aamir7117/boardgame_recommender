import re
from math import ceil
from collections import defaultdict
import pandas as pd
import graphlab
import sqlite3
from tabulate import tabulate
from pymongo import MongoClient
# import sklearn
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
import random
import gc
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

client = MongoClient()
db = client['bgg_data']
games = db['games']
all_info = db['all_info']
master = db['master']


def machine_counts():
    machine_1 = db['machine_1']
    machine_2 = db['machine_2']
    machine_3 = db['machine_3']
    machine_4 = db['machine_4']
    machine_5 = db['machine_5']
    machine_6 = db['machine_6']
    machine_7 = db['machine_7']
    machine_8 = db['machine_8']
    machine_9 = db['machine_9']
    machine_10 = db['machine_10']
    machine_11 = db['machine_11']
    machine_12 = db['machine_12']
    machine_13 = db['machine_13']
    machine_14 = db['machine_14']
    machine_15 = db['machine_15']
    machine_16 = db['machine_16']
    machine_17 = db['machine_17']
    machine_18 = db['machine_18']
    machine_19 = db['machine_19']
    machine_20 = db['machine_20']
    machine_21 = db['machine_21']
    machine_22 = db['machine_22']
    machine_23 = db['machine_23']
    machine_24 = db['machine_24']
    machine_25 = db['machine_25']
    machine_26 = db['machine_26']
    machine_27 = db['machine_27']
    machine_28 = db['machine_28']
    machine_29 = db['machine_29']
    machine_30 = db['machine_30']
    machine_31 = db['machine_31']
    machine_32 = db['machine_32']

    machines = [machine_1, machine_2, machine_3, machine_4, machine_5, machine_6, machine_7, machine_8, machine_9, machine_10, machine_11, machine_12, machine_13, machine_14, machine_15, machine_16, machine_17, machine_18, machine_19, machine_20, machine_21, machine_22, machine_23, machine_24, machine_25, machine_26, machine_27, machine_28, machine_29, machine_30, machine_31, machine_32]
    return [ratings.count() for ratings in machines]

# print machine_counts()


# def concat_df(df):
#     '''only use if imported each machine's data into a pd.df and then you want to concat'''
#     cols = ['game_id','game_name','player_name','player_rating']
#     df = pd.DataFrame(list(machine_1.find()),columns=cols)
#     print "done with machine_1"
#     for number, machine in enumerate(machines[1:],2):
#         df_temp = pd.DataFrame(list(machine.find()),columns=cols)
#         df = df.append(df_temp)
#         print "done with machine_{}".format(number)
#     return df

# df = concat_data()

def make_pandas_df(mongodb_collection):
    cols = ['game_id','game_name','player_name','player_rating']
    df = pd.DataFrame(list(mongodb_collection.find()),columns=cols)
    return df

def split_data(df,num_ratings):
    '''
    df, holdout = split_data(df,num_ratings)
    where num_ratings are the number of ratings you want to remove from df.
    The ratings removed will be from users with more than one rating so to not
    create a cold start situation.
    Will be optimized in the future with parrallel computing.
    Use recursively to get as many test, holdout sets as needed.
    df_train, df_test = split_data(df,int(len(df)*0.2))  #20 \% train test split
    '''
    counts = df['player_name'].value_counts()
    active_users = counts[counts>1] #let's not pickout a user that has only one rating since that would turn this into a coldstart problem
    holdout = pd.DataFrame(columns=df.columns)
    # indexes_to_remove = []
    for rating in xrange(num_ratings):
        print "working on {} out of {}".format(rating,num_ratings-1)
        index = random.randint(0,len(df)-1)
        rating = df.iloc[index] #initialize rating variable with a random grab from df
        while rating['player_name'] not in active_users.index:
            index = random.randint(0,len(df)-1)
            rating = df.iloc[index]
        holdout = holdout.append(rating)
        df = df.drop(index)
        gc.collect()
        # del df
        # indexes_to_remove.append(index)

    del active_users
    return df, holdout

def make_svd_recommender(df):
    '''runs the graphlab recommender and returns the '''
    turi_SFrame = graphlab.SFrame(df[['game_id','player_name','player_rating']])
    recommender = graphlab.recommender.factorization_recommender.create(turi_SFrame,
                                                      user_id = 'player_name',
                                                     item_id = 'game_id',
                                                     target = 'player_rating',
                                                     solver = 'als')
    return recommender

def load_boardgames():
    '''returns two dataset boardgames, expansions'''
    conn = sqlite3.connect('database.sqlite')  #database.sqlite file needs to be in current folder
    boardgames_all = pd.read_sql("SELECT * FROM BoardGames;",conn)

    removals = ['details.image','details.thumbnail','attributes.boardgamecompilation','attributes.boardgameexpansion','attributes.boardgameimplementation','attributes.boardgameintegration','stats.family.abstracts.bayesaverage','stats.family.abstracts.pos','stats.family.cgs.bayesaverage','stats.family.cgs.pos','stats.family.childrensgames.bayesaverage','stats.family.childrensgames.pos','stats.family.familygames.bayesaverage','stats.family.familygames.pos','stats.family.partygames.bayesaverage','stats.family.partygames.pos','stats.family.thematic.bayesaverage','stats.family.thematic.pos','stats.family.wargames.bayesaverage','stats.family.wargames.pos','stats.median','polls.suggested_numplayers.10','polls.suggested_numplayers.5','polls.suggested_numplayers.6','polls.suggested_numplayers.7','polls.suggested_numplayers.8','polls.suggested_numplayers.9','stats.family.amiga.bayesaverage','stats.family.amiga.pos','stats.family.arcade.bayesaverage','stats.family.arcade.pos','stats.family.atarist.bayesaverage','stats.family.atarist.pos','stats.family.commodore64.bayesaverage','stats.family.commodore64.pos','stats.subtype.rpgitem.bayesaverage','stats.subtype.rpgitem.pos','stats.family.strategygames.bayesaverage','stats.family.strategygames.pos','stats.subtype.boardgame.bayesaverage', 'stats.subtype.boardgame.pos','polls.language_dependence','attributes.t.links.concat.2....','polls.suggested_playerage','stats.subtype.videogame.bayesaverage','stats.subtype.videogame.pos', 'attributes.boardgameartist','attributes.boardgamefamily','attributes.boardgamepublisher','attributes.boardgamefamily','attributes.boardgamepublisher','details.minplaytime','stats.averageweight']

    boardgames_all.drop(removals,axis=1,inplace=True)
    boardgames_all = boardgames_all[boardgames_all['stats.usersrated']!=0] # user rating is my response variable so I don't want games that haven't been rated!
    expansions = boardgames_all[boardgames_all['game.type']=='boardgameexpansion']
    expansions.drop('game.type',axis=1,inplace=True)
    boardgames = boardgames_all[boardgames_all['game.type']=='boardgame']
    boardgames.drop('game.type',axis=1,inplace=True)
    boardgames.set_index('game.id',inplace=True)
    del boardgames_all
    return boardgames,expansions

def print_boardgames(pd_dataframe,rows=5):
    for i in xrange(1,len(pd_dataframe.columns)-3, 4):
        print pd_dataframe.iloc[:,i:i+4].head(rows)


def binarize_categories(boardgames,nc=8):
    pool = Pool(nc) #num_cores
    boardgames['attributes.boardgamecategory'] = pool.map(str,boardgames['attributes.boardgamecategory'])
    unique_mechanics = set() #categories
    for mechanics in boardgames['attributes.boardgamecategory']:
        split = mechanics.split(',')
        for mechanic in split:
            unique_mechanics.add(mechanic)
    for mechanic in unique_mechanics:
        boardgames[mechanic] = 0  #initialize a new column for each mechanice
    # for column in boardgames:
    #     for row in column:
    #         row[mechanic] = 1
    len(boardgames['attributes.boardgamecategory'])
    for i, mechanics in enumerate(boardgames['attributes.boardgamecategory']):
        split = mechanics.split(',')
        if len(split):
            for spl in split:
                boardgames.iloc[i,:][spl] = 1

def maxplayers_refine(bd_light):
    '''bd_light should be a boardgames dataset with a column labeled details.maxplayers.
    This function replaces entries below 1 and over 200 with the average of the entries in this range'''
    mask = (bd_light['details.maxplayers']<200) & (bd_light['details.maxplayers']>=1)
    median = bd_light['details.maxplayers'][mask].median()
    bd_light['details.maxplayers'][np.invert(mask)] = median # set all 24 outliers to median

def minage_refine(bd_light):
    '''
    this function replaces entries >= 18 to 3, below 5 to 0,  11 <= x >= 5 --> 1, 11 < x <18 -->2
    '''
    bd_light['details.minage'][bd_light['details.minage']<5]=0
    bd_light['details.minage'][(bd_light['details.minage']>=5) & (bd_light['details.minage']<=11)]=1
    bd_light['details.minage'][(bd_light['details.minage']>11) & (bd_light['details.minage']<18)]=2
    bd_light['details.minage'][bd_light['details.minage']>=18]=3

def minplayers_refine(bd_light):
    ''' this function caps minplayers to 10 so any entries over 10 become 10. Also any entries less than or
    equal to 0 become 1'''
    bd_light['details.minplayers'][bd_light['details.minplayers']>10] = 10
    bd_light['details.minplayers'][bd_light['details.minplayers']<=0] = 1

def playingtime_refine(bd_light):
    '''this function will bucketize playtimes as follows:
    x > 360 --> 360
    x >120 & x<=360 --> 120
    x <=120 & x>=60 --> 60
    x <60 & x>=30 --> 30
    x <30 & x>=15 --> 15
    x <15 & x>=5 --> 5
    x <5 --> 0 '''

    bd_light['details.playingtime'][bd_light['details.playingtime']>360]=360
    bd_light['details.playingtime'][(bd_light['details.playingtime']<360)&(bd_light['details.playingtime']>=120)]=120
    bd_light['details.playingtime'][(bd_light['details.playingtime']<120)&(bd_light['details.playingtime']>=60)]=60
    bd_light['details.playingtime'][(bd_light['details.playingtime']<60)&(bd_light['details.playingtime']>=30)]=30
    bd_light['details.playingtime'][(bd_light['details.playingtime']<30)&(bd_light['details.playingtime']>=15)]=15
    bd_light['details.playingtime'][(bd_light['details.playingtime']<15)&(bd_light['details.playingtime']>=5)]=5
    bd_light['details.playingtime'][bd_light['details.playingtime']<5]=0

def yearpublished_refine(bd_light):
    '''this function replaces <1900 & >2017 with median year of this range'''
    mask = (bd_light['details.yearpublished']>=1900) & (bd_light['details.yearpublished']<=2017)
    median_year = bd_light['details.yearpublished'][mask].median()
    bd_light['details.yearpublished'][np.invert(mask)] = median_year


def calc_avg_rmse(predictions,SFrame):
    return np.average((predictions - SFrame)**2)**0.5

def compare_rmse(sf_train,sf_test,test_size=0.11):
    df_train, df_test = train_test_split(df,test_size=test_size)
    sf_train = graphlab.SFrame(df_train)
    sf_test = graphlab.SFrame(df_test)
    recommender = graphlab.recommender.factorization_recommender.create(sf_train,\
                                                                        user_id='player_name', \
                                                                        item_id='game_id',\
                                                                        target='player_rating'\
                                                                        ,solver='als')
    evaluation = recommender.evaluate(sf_test)
    baseline = calc_avg_rmse(sf_train['player_rating'].mean(), sf_test['player_rating'])
    return evaluation['rmse_overall'], baseline

'''
Observations:

1. the 'attributes.boardgamecategory' column has several categories per game separated by commas. I will binarize for the unique 85 categories.

'''

# if __name__ == '__main__':
#     print "making pandas dataframe with master collection"
#     df = make_pandas_df(master)  #assuming mongodb collection master has all the rating records
#     print "splitting data into holdout and df"
#     df, holdout = split_data(df,50) #let's make a holdout set of 50 ratings from users that have more than one rating
#     print "splitting data into df_train an df_test"
#     df_train, df_test = split_data(df, 20)
#     print "making sf_train (turi SFrame)"
#     sf_train = graphlab.SFrame(df_train)
#     print "making sf_test (turi SFrame)"
#     sf_test = graphlab.SFrame(df_test)
#     print "making sf_holdout (turi SFrame)"
#     sf_test = graphlab.SFrame(holdout)
#     print "shuffling the SFrame, sf_train"
#     sf_train = graphlab.cross_validation.shuffle(sf_train)
#     gc.collect()
    # print "making factorization recommender"
    # recommender = graphlab.recommender.factorization_recommender.create(sf_train,\
    #                                                                     user_id='player_name', \
    #                                                                     item_id='game_id',\
    #                                                                     target='player_rating'\
    #                                                                     ,solver='als')
    # #the above recommender with default params takes approximately 1min 15s on a 30G RAM, 8 core AWS EC2 instance.
    #
    # print "recommender's training rmse is {}".format(recommender.training_rmse) # 0.9764571085111885
    # evaluation = recommender.evaluate(sf_test)
    # print "recommender's validation error is {}".format(evaluation['rmse_overall'])  # 0.8260685832241673
    # # looks like it is systematically picking up error
    #
    # baseline_rmse = calc_avg_rmse(sf_train['player_rating'].mean(), sf_test['player_rating'])
    #
    #
    # print "loading boardgames item_info dataset"
    # boardgames, expansions = load_boardgames()
    # del expansions
    # cols = ['details.maxplayers','details.minage','details.minplayers','details.playingtime','details.yearpublished','stats.average','stats.numcomments','stats.owned']
    #
    # print "making bd_light, a few selected columns of the boardgames dataset"
    # print "refining columns of bd_light"
    # bd_light = boardgames[cols]
    # maxplayers_refine(bd_light)
    # minage_refine(bd_light)
    # minplayers_refine(bd_light)
    # playingtime_refine(bd_light)
    # yearpublished_refine(bd_light)
    # bd_light = bd_light.fillna(bd_light.median(axis=0))
    #
    # '''  # Did some linear regression and random forest to try and identify which columns to keep
    #
    # y = np.array(bd_light['stats.average'])
    # x_prep = bd_light.drop('stats.average',axis=1)
    # X = np.array(x_prep)
    # '''
    #
    # '''
    # LR = LinearRegression()
    # X_train, X_test, y_train, y_test = train_test_split(X,y)
    # print "Regressing with Linear model onto stats.average"
    # LR.fit(X_train,y_train)
    # print "Test score for model with 25% test_size, r^2 = {}".format(LR.score(X_test,y_test)) # r^2 = 0.1274,  Ridge was similar
    # for item in sorted(zip(x_bd_light.columns,LR.coef_),key=lambda x:x[1],reverse=True):
    #     print item
    #     # it seems cols minage,yearpub,playingtime are important contributors
    # '''
    #
    # '''
    # rfr = RandomForestRegressor()
    # print "Regressing with RandomForestRegressor onto stats.average"
    # rfr.fit(X_train,y_train)
    # print "RandomForest regressor r^2 = {}".format(rfr.score(X_test,y_test)) # r^2 = 0.14698
    # for item in sorted(zip(x_bd_light.columns,rfr.feature_importances_),key=lambda x:x[1],reverse=True):
    #     print item
    #     # it seems cols yearpub, numcomments, owned, playingtime and minage are important
    # '''
    #
    # keep_cols = ['details.minage','details.yearpublished','details.playingtime']
    # item_info = bd_light[keep_cols]
    # item_info[keep_cols] = MinMaxScaler().fit_transform(item_info[keep_cols])
    # item_info = item_info.reset_index()
    # item_info = graphlab.SFrame(item_info)
    # item_info.rename({'game.id':'item_id'})
    # item_info['item_id'] = item_info['item_id'].astype(int) #ratings sf has game_id as int
    # gc.collect()
    # num_folds = 10
    # folds = graphlab.cross_validation.KFold(sf_train,num_folds)
    #
    # # Without side_data
    # param_no_side_data = dict(user_id='player_name', item_id='game_id',target='player_rating',solver='als')
    # print "{}-fold validation without side_data running in the background. This will take a while".format(num_folds)
    # job_no_side_data = graphlab.cross_validation.cross_val_score(folds,graphlab.recommender.factorization_recommender.create,param_no_side_data)
    # results_no_sd = job_no_side_data.get_results()
    # print "no_side_data training_rsme is {}".format(np.average(results_no_sd['summary']['training_rmse'])) # 0.963752221314
    # print "no_side_data validation_rsme is {}".format(np.average(results_no_sd['summary']['validation_rmse'])) # 1.58021418202
    # # recall_no_sd = np.average(results_no_sd['summary']['validation_recall@5'])
    # # precision_no_sd = np.average(results_no_sd['summary']['validation_precision@5'])
    # # print "validation recall = {:0.7f}, precision = {:0.7f} respectively\nwhere 5 recommendations were made to each user. No side data was used.".format(recall_no_sd, precision_no_sd)
    #
    # # With side_data
    # params_with_side_data = dict(user_id='player_name', item_id='game_id',target='player_rating',solver='als',item_data=item_info)
    # print "{}-fold validation WITH side_data running in the background. This will take a while".format(num_folds)
    # job_with_side_data = graphlab.cross_validation.cross_val_score(folds,graphlab.recommender.factorization_recommender.create,params_with_side_data)
    # results_sd = job_with_side_data.get_results()
    # print "With side_data, the training_rsme is {}".format(np.average(results_sd['summary']['training_rmse']))  # 0.963580729466
    # print "With side_data, the validation_rsme is {}".format(np.average(results_sd['summary']['validation_rmse'])) # 1.57877848137
    # # recall_sd = np.average(results_sd['summary']['validation_recall@5'])
    # # precision_sd = np.average(results_sd['summary']['validation_precision@5'])
    # # print "validation recall = {:0.7f}, precision = {:0.7f} respectively\nwhere 5 recommendations were made to each user. Side data was used".format(recall_sd, precision_sd)
    #
    #
    # recommender_1 = results_sd['models'][0]
    # user_1 = 'mdaffonso' # an existing user from the input data
    # num_recs = 10
    # print "printing {} recommendations for user: {}".format(num_recs,user_1)
    # print recommender_1.recommend([user_1],num_recs) # make 10 recommendations for user_1
    #
    #
    # # Tried different dimensions for U and V matrices but no improvements
    # # ks = [   8,   24,   72]
    # # for k in ks:
    # #     k_models.append(graphlab.recommender.factorization_recommender.create(sf_train,user_id ='player_name', item_id='game_id',target='player_rating',solver='als', num_factors=k))
    # # for number, model in enumerate(k_models, 1):
    # #     evaluation = recommender.evaluate(sf_test)
    # #     print "model {} results\ntest_rmse={}\n\n".format(number, evaluation['rmse_overall'])
    # #
    # #     # model 1 results
    # #     # test_rmse=0.826068583224
    # #     #
    # #     # model 2 results
    # #     # test_rmse=0.826068583224
    # #
    # #     # model 3 results
    # #     # test_rmse=0.8260685832241673
    #
    #
    #
    # # 8 columns_side_data - all of bd_light as side_data
    # item_info = bd_light
    # item_info[item_info.columns] = MinMaxScaler().fit_transform(item_info[item_info.columns])
    # item_info = item_info.reset_index()
    # item_info = graphlab.SFrame(item_info)
    # item_info.rename({'game.id':'item_id'})
    # item_info['item_id'] = item_info['item_id'].astype(int) #ratings sf has game_id as int
    # gc.collect()
    # num_folds = 10
    # folds = graphlab.cross_validation.KFold(sf_train,num_folds)
    # big_job_with_side_data = graphlab.cross_validation.cross_val_score(folds,graphlab.recommender.factorization_recommender.create,params_with_side_data)
    #
    #
    # results_big_sd = big_job_with_side_data.get_results()
    # print "With side_data, the training_rsme is {}".format(np.average(results_big_sd['summary']['training_rmse'])) # 0.96366534225
    # print "With side_data, the validation_rsme is {}".format(np.average(results_big_sd['summary']['validation_rmse'])) # 1.58118597747

    #performs the same as baseline. No improvements

    # cutoffs = [10,50,100]
    # model1 = results_big_sd['models'][0]
    # eva = model1.evaluate_precision_recall(sf_test,cutoffs=cutoffs) #trying different k values
    # print eva['precision_recall_by_user'].print_rows(40)


import pandas as pd
from pymongo import MongoClient
import numpy as np
import random
import gc
from multiprocessing import Pool
import graphlab
from collections import defaultdict


client = MongoClient()
db = client['bgg_data']
games = db['games']
all_info = db['all_info']
master = db['master']

def make_pandas_df(mongodb_collection):
    cols = ['game_id','game_name','player_name','player_rating']
    df = pd.DataFrame(list(mongodb_collection.find()),columns=cols)
    return df

min_like_threshold = 5.0
holdout_ratings = 50
test_ratings = 20
print "making pandas dataframe with master collection"
df = make_pandas_df(master)  #assuming mongodb collection master has all the rating records


def randomly_split_data(df):
    print "splitting data into holdout and df"
    df, holdout = split_data(df,holdout_ratings) #let's make a holdout set of 50 ratings from users that have more than one rating
    print "splitting data into df_train an df_test"
    df_train, df_test = split_data(df, test_ratings)
    print "making sf_train (turi SFrame)"
    sf_train = graphlab.SFrame(df_train)
    print "making sf_test (turi SFrame)"
    sf_test = graphlab.SFrame(df_test)
    print "making sf_holdout (turi SFrame)"
    sf_holdout = graphlab.SFrame(holdout)
    print "shuffling the SFrame, sf_train"
    sf_train = graphlab.cross_validation.shuffle(sf_train)
    gc.collect()
    sf_train_stripped = sf_train[['game_id','game_name','player_name']]
    sf_test_stripped = sf_test[['game_id','game_name','player_name']]





#All the next few models are ITEM to ITEM similarity models

def make_cosine_similarity_model():
    # Make the cosine similarity model
    print "making cosine similarity model"
    cos_sim_model = graphlab.recommender.item_similarity_recommender.create(sf_train, user_id='player_name', item_id='game_id',target='player_rating',similarity_type='cosine')
    print "evaluating model"
    eval_cos_sim = cos_sim_model.evaluate_rmse(sf_test,target='player_rating')
    print "average_rmse_per_user = {}".format(np.average(eval_cos_sim['rmse_by_user']['rmse']))  # 6.89690434713
    print "average_rmse_per_item = {}".format(np.average(eval_cos_sim['rmse_by_item']['rmse'])) # 6.88305804898
    print "average_rmse_overall = {}".format(np.average(eval_cos_sim['rmse_overall'])) # 7.06256468879
    '''
    average_rmse_per_user = 6.89690434713
    average_rmse_per_item = 6.88305804898
    average_rmse_overall = 7.06256468879
    # rmse overall is every single point in the test set - predicted squared ....
    '''

    prec_rec_cos_sim = cos_sim_model.evaluate(sf_test_stripped,metric='precision_recall',cutoffs=[10,50,100,500])
    print prec_rec_cos_sim['precision_recall_overall']
    '''
    Precision and recall summary statistics by cutoff
    +--------+----------------+-------------+
    | cutoff | mean_precision | mean_recall |
    +--------+----------------+-------------+
    |   10   |     0.004      |     0.04    |
    |   50   |     0.0008     |     0.04    |
    |  100   |     0.0006     |     0.06    |
    |  500   |    0.00024     |     0.12    |
    +--------+----------------+-------------+
    '''



def make_jaccard_similarity_model():
    # Make the jaccard similarity model
    print "making jaccard similarity model"
    jac_sim_model = graphlab.recommender.item_similarity_recommender.create(sf_train, user_id='player_name', item_id='game_id',target='player_rating',similarity_type='jaccard')
    print "evaluating model"
    eval_jac_sim = jac_sim_model.evaluate_rmse(sf_test,target='player_rating')
    print "average_rmse_per_user = {}".format(np.average(eval_jac_sim['rmse_by_user']['rmse'])) # 7.59259646544
    print "average_rmse_per_item = {}".format(np.average(eval_jac_sim['rmse_by_item']['rmse'])) # 6.69348524486
    print "average_rmse_overall = {}".format(np.average(eval_jac_sim['rmse_overall'])) # 7.1366255703
    '''
    average_rmse_per_user = 7.08953561495
    average_rmse_per_item = 7.05410613542
    average_rmse_overall = 7.26437895187
    # rmse overall is every single point in the test set - predicted squared ....
    '''
    jac_sim_model = graphlab.recommender.item_similarity_recommender.create(sf_train_stripped, user_id='player_name', item_id='game_id',similarity_type='jaccard')
    prec_rec_jac_sim = jac_sim_model.evaluate(sf_test_stripped,metric='precision_recall',cutoffs=[10,50,100,500])
    print prec_rec_jac_sim['precision_recall_overall']
    '''
    Precision and recall summary statistics by cutoff
    +--------+----------------+-------------+
    | cutoff | mean_precision | mean_recall |
    +--------+----------------+-------------+
    |   10   |      0.0       |     0.0     |
    |   50   |     0.0008     |     0.04    |
    |  100   |     0.0008     |     0.08    |
    |  500   |    0.00024     |     0.12    |
    +--------+----------------+-------------+
    '''



def make_pearson_similarity_model():
    # Make the pearson similarity model
    print "making pearson similarity model"
    pearson_sim_model = graphlab.recommender.item_similarity_recommender.create(sf_train, user_id='player_name', item_id='game_id',target='player_rating',similarity_type='pearson')
    print "evaluating model"
    eval_pearson_sim = pearson_sim_model.evaluate_rmse(sf_test,target='player_rating')
    print "average_rmse_per_user = {}".format(np.average(eval_pearson_sim['rmse_by_user']['rmse']))
    print "average_rmse_per_item = {}".format(np.average(eval_pearson_sim['rmse_by_item']['rmse']))
    print "average_rmse_overall = {}".format(np.average(eval_pearson_sim['rmse_overall']))
    '''
    average_rmse_per_user = 1.21355464965
    average_rmse_per_item = 1.21355464965
    average_rmse_overall = 1.72022286641
    # rmse overall is every single point in the test set - predicted squared ....
    '''

    prec_rec_pearson = pearson_sim_model.evaluate(sf_test_stripped,metric='precision_recall',cutoffs=[10,50,100,500])
    print prec_rec_pearson['precision_recall_overall']
    '''
    Precision and recall summary statistics by cutoff
    +--------+----------------+-------------+
    | cutoff | mean_precision | mean_recall |
    +--------+----------------+-------------+
    |   50   |      0.0       |     0.0     |
    |   10   |      0.0       |     0.0     |
    |  100   |      0.0       |     0.0     |
    |  500   |     4e-05      |     0.02    |
    +--------+----------------+-------------+
    '''
    # check with rating threshold
    prec_rec_pearson = pearson_sim_model.evaluate(sf_test[sf_test['player_rating']>5.0],metric='precision_recall',cutoffs=[10,50,100,500])
    print "with minimum like threshold of {}".format(min_like_threshold)
    print prec_rec_pearson['precision_recall_overall']
    '''
    +--------+------------------+-----------------+
    | cutoff |    precision     |      recall     |
    +--------+------------------+-----------------+
    |   10   |       0.0        |       0.0       |
    |   50   |       0.0        |       0.0       |
    |  100   |       0.0        |       0.0       |
    |  500   | 4.6511627907e-05 | 0.0232558139535 |
    +--------+------------------+-----------------+
    '''

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
