
Problem:
Boardgamegeek.com is a grassroots company in the online boardgame community. Scott Alden and Derk Solko, two software engineers and avid boardgame players, founded the company in the year 2000 and continue to actively manage it to this day with a very minimal staff. This website has a very strong community, has received several accolades acknowledging it as a one of a kind resource for boardgame enthusiasts all over the world. It generates approximately 21 million unique visitors every year.

The main function that the website serves is boardgame ranking which is mainly used by existing users to get a sense of which game to buy next.

Data:
1. 4.625760 million ratings on said boardgames. 163,285 users, matrix sparsity = 0.005685
2. 4983 unique boardgames with about 5 useful attributes (before feature engineering)

Model:
Funk-SVD. Cross_val_score without side_data = 1.59 (3 folds with about (0.33 test_size)
10-fold model with default features. --> training rmse = 0.963, test_rmse = 1.59
https://turi.com/products/create/docs/generated/graphlab.recommender.ranking_factorization_recommender.RankingFactorizationRecommender.html #see the formula for side_data here.





Side data including:
      a. boardgame mechanics (strategy, adventure, farming etc.)
      b. NLP on the rich description of each boardgame (might not give me any signal I think)
      c. maybe I'll do NMF to extract groups of features of boardgames and make my questions to a cold start user based on these groups....


Future:
Try implicit recommender, ratings not included but collections filtered on some threshold of rating for each user
Do not recommend games already in collection (not just rated) --scrape this data first.
run LeaveOneOut cross val. It would take 10+ years on a 30G 8-core EC2 instance so bigger comp perhaps
make a webapp where an existing user can enter his/her username and get top 10 recommendations with predicted ratings for each.




Notes:
My train test split is biased in that it only selects users that have more than one rating to put in the test set. I have it this way so it doesn't create a cold start situation as this model is intended for predicting ratings for users that have already made at least one rating.


Below is a list of the attributes I'm cleaning. Probably going to drop a few of them and binarize some..

  <class 'pandas.core.frame.DataFrame'>
Index: 76688 entries, 1 to 220070
Data columns (total 30 columns):
row_names                          76688 non-null object
details.maxplayers                 76685 non-null float64     replace 5269 entries (<=0), 27 (>200) with average, IQR?
details.maxplaytime                76685 non-null float64     replace ~18k entried (<4)
details.minage                     76685 non-null float64     bucketize (<=0 replace, 0<x<3 feat_1, 3<x<7 feat_2, 7-15-->f3 f4 = 15+)
details.minplayers                 76685 non-null float64     replace <=0 with 1, maybe remove games with min_players >4 from recommendation (only 668 games which is 0.8%)
details.minplaytime                76685 non-null float64     replace <=0 with 1
details.playingtime                76685 non-null float64     replace <=0 with 1
details.yearpublished              76685 non-null float64     maybe replace <1900 with avg  &  >2017 with avg
attributes.boardgamecategory       75297 non-null object      binarize! split into features (has 1391 isnull() )
attributes.boardgamedesigner       67101 non-null object      designed > 10 games --> feature
attributes.boardgamefamily         39776 non-null object      drop this column
attributes.boardgamemechanic       62971 non-null object      binarize! the unique 85 categories
attributes.boardgamepublisher      76593 non-null object      drop this column
attributes.total                   76688 non-null float64
stats.average                      76688 non-null float64
stats.averageweight                76688 non-null float64     drop this column. Unfortunately the metric is defined poorly and I believe people's interpretation of it are vastly different.
stats.bayesaverage                 76688 non-null float64     
stats.numcomments                  76688 non-null float64
stats.numweights                   76688 non-null float64
stats.owned                        76688 non-null float64     
stats.stddev                       76688 non-null float64
stats.trading                      76688 non-null float64
stats.usersrated                   76688 non-null float64     filter out all rows where this is = 0  (22992 rows)
stats.wanting                      76688 non-null float64
stats.wishing                      76688 non-null float64
polls.suggested_numplayers.1       15148 non-null object
polls.suggested_numplayers.2       17712 non-null object
polls.suggested_numplayers.3       12781 non-null object
polls.suggested_numplayers.4       13058 non-null object
polls.suggested_numplayers.Over    13899 non-null object
dtypes: float64(19), object(11)
memory usage: 20.6+ MB
