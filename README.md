## Boardgame Recommender

### Motivation

I love boardgames and every time I come across a new one that I like, I think to myself: how come I haven't heard of this game before? I know there are many boardgame fans out there with the same dilemma. In fact, lots of these fans hang out on boardgamegeek.com where they rate games and can see the average rating of a lot of games. Okay, that helps to know which games are the most popular based on their average ratings. But with all of this data, we should be able to make a better recommendation. So that's what I did.

### Data
I used 10 parallelized Selenium scrapers to extract the individual ratings for every boardgame that was rated on boardgamegeek.com for the last 17 years, across the whole world. In total, I have the ratings of 165,000 users across 5000 boardgames. This data is very sparse (1/200 values present) and naturally, some boardgames have many more ratings than others. Users also follow this pattern, in that, some users rate much more frequently than others.

### Framework
I know from personal experience that not all boardgame fans are the same. My sister for example really likes games like Codenames and Bananagrams but she isn't really interested in games like Dominion, 7-wonders, Specter Ops etc. So I knew I had to subset my 165,000 users into a Casual player or an Advanced player. I chose users 1-50 ratings as casual/light users and >50 ratings as advanced users. This was convenient because around 60% of users have rated at least 50 games so both user subsets had a substantial amount of users in them. Also, the variance in ratings increased from xxxx for all users to xxxx for just advanced users. For light users it dropped to xxxx. This indicates that advanced users tend to rate further away from the average and are perhaps more critical of the games. 

### Recommender
To my casual user, I'd like to recommend more popular boardgames like Catan, Carcasonne and Banagrams but to my advanced user, I want to take advantage of the finer grain differences in their ratings so I used a Pearson similarity model. Pearson similarity is essentially the same as a cosine similarity adjusted for the mean rating of each game (see equation 1). This worked really well in accentuating the minor differences from the mean, which is quite meaningful for advanced users. 

For casual users, I used a Jaccard and Cosine similarity metric, both of which performed similarly in RMSE. This indicates that the fact that a casual user has rated a particular game contributes more to the predicted rating than the values of the rating itself! So my final model for this user subset uses Jaccard similarity. 

There is much more to be done on this dataset including joining it with game data, using features like playing time, mechanics (dice roll, cards, strategy) etc.

I have already tried a funk-SVD model for my advanced users with game features as side features. However, for the sake of the first recommender, since none exist in the market already, a simple approach like jaccard & pearson would be a smart first pass. After validating with these models using RMSE and even precision/recall as a metric, I would then try the funk SVD model.

Feel free to connect with me if you'd like to chat more about this project.
