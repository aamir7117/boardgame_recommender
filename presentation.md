To do list:

1. Why are boardgames important?
describe how fun they are. Name a few in order of increasing complexity.

2. Why is recommending games important?
active boardgame community, 2 x $25 x 100,000 = $5 million/year
No models being used currently. Rankings based on average ratings.

3. What makes users similar?
   What makes boardgames similar?

With these thoughts, I sought to make a boardgame recommender

So I scraped the most popular website for boardgames there is!
  Boardgamegeek-users. Basic website but very active community.
  Parallel Selenium scrapers to extract 0.5 billion ratings for
  5000 boardgames

Made by games vs users matrix
1/200 sparse

Noticed the distribution of ratings:
GRAPH
Oh, most ratings are between 4 and 8.

What to do about that?


Item-Item cosine similarity recommender
RMSE? Precision, recall? what does that mean?? Compare with simple mean recommendation.
Not great bc doesn't account for mean or variance of the user row

Baseline model: recommend uniforml (k, precision, recall):
10 0.0315342837746 0.00725358450056
50 0.0283047050037 0.0552081196677
100 0.0267947586394 0.106985852733
500 0.0176792721841 0.320059292351



Then I tried Funk-SVD (Netflix 1MM prize winning model)
Precision, recall? Oooh, not so good!

Let's try with some side_data:
features X,Y,Z extracted also from boardgamegeek. Fortunately didn't have to scrape this info.
Formula --> show where side_data goes, show cost function, explain idea of how it works.
Precision, recall? Still no good.
Conjecture, in this sea of dissimilar users (some who vote once, other who vote 50+ games),
it is hard for Funk SVD to fill in the gaps accurately because the averages get drowned out.

Future:
Separate the data into two groups --> active members (10+ ratings)
Non-active members --> rest

Cold-start?
Tell us your 5 favorite boardgames?
input into the model --> predictions.
Have more than 10? you go into the advanced model.
