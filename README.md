## Boardgame Recommender

### Motivation

I love boardgames but how do I select which one to buy next? There are many boardgame fans out there with the same dilemma. In fact, lots of these fans hang out on boardgamegeek.com where they rate games. Since only the aggregate ratings are displayed, only the most popular ones rise to the top. But with all of the individual ratings, we should be able to make a better recommendation. So that's what I did.

### Data
I used 10 parallelized Selenium scrapers to extract the individual ratings for every boardgame that was rated on boardgamegeek.com for the last 17 years. In total, I have the ratings of 165,000 users across 5000 boardgames. This data is very sparse (1/200 values present) and naturally, some boardgames have many more ratings than others. Users also follow this pattern, in that, some users rate much more frequently than others.

### Thought process
I know from personal experience that not all boardgame fans are the same. My sister for example really likes games like Codenames and Bananagrams but she isn't really interested in games like Dominion, 7-wonders, Specter Ops etc. So I knew I had to subset my 165,000 users into a Casual player or an Advanced player. I chose users with 1-50 ratings as casual/light users and >50 ratings as advanced users. This was convenient because around 60% of users have rated at least 50 games so both user subsets had a substantial amount of data. Also, the baseline RMSE for light users was lower than for all users combined and higher for advanced users. This indicates that advanced users tend to rate further away from the average and are perhaps more critical of the games. 

### Recommender
To my casual user, I wanted to recommend more popular boardgames like Catan, Carcasonne and Banagrams. To my advanced user, I wanted to recommend advanced games so I used a Pearson similarity. Pearson similarity is essentially the same as cosine similarity adjusted for the mean rating of each game. This worked really well in accentuating the minor differences from the mean, which is how the complex recommendations are generated.

For casual users, Jaccard and Cosine similarity metric both performed similarly in RMSE. This indicates that the fact that a casual user has rated a particular game contributes more to the predicted rating than the value of the rating itself! This makes sense from a business perspective because if a light user has taken time to rate a boardgame, he/she probably liked the game to begin with. Light users tend to buy games that they have played elsewhere and enjoyed already instead of buying new games they haven't yet played.

I also made a Funk-SVD model (Netflix 1MM prize winner) for my advanced users with side data (playing time, # players, suggestedd age etc). However, for simplicity's sake, I would prefer to start with the Pearson model instead of Funk-SVD. Only after validating the simpler models, would I want to employ the Funk-SVD model. I would validate with precision not F1 since we can't ask users to rate all 5000 games.
