## Boardgame Recommender

### Motivation

I love boardgames but how do I select which one to buy next? There are many boardgame fans out there with the same dilemma. In fact, lots of these fans hang out on boardgamegeek.com where they rate games. Since only the aggregate ratings are displayed, only the most popular ones rise to the top. But with all of the individual ratings, we should be able to make a better recommendation. So that's what I did.

### Data
This data is not pubicly available and was quite hard to get a hold of. I used 10 parallel Selenium scrapers to extract the individual ratings for every boardgame on boardgamegeek.com for the last 17 years. In total, I have ratings for 165,000 users across 5000 boardgames. This data is very sparse (1/200 values present) and naturally, some boardgames have more ratings than others. Users also follow this pattern, in that, some users rate much more frequently than others.

### Thought process
I know from personal experience that boardgame fans come in varieties. My sister for example really likes games like Codenames and Bananagrams but she doesn't like more complicated games Dominion, 7-wonders, Specter Ops etc. Extrapolating this logic, I subset my 165,000 users into Casual users and Advanced users. I chose users with 1-50 ratings as casual users and >50 ratings as advanced users. This was convenient because around 60% of users have rated at least 50 games so both user subsets had a substantial amount of data. Also, the baseline RMSE for light users was lower than for all users combined and higher for advanced users. This indicates that advanced users tend to rate further away from the average and are perhaps more critical of the games. 

### Recommender
To my casual user, I wanted to recommend more popular boardgames like Catan, Carcasonne and Banagrams so I used Jaccard and Cosine similarities. Both models performed similarly in RMSE. This indicates that the prescence of the rating contributes more to the predicted value than the value of the rating itself! This makes sense from a business perspective because if a light user has taken time to rate a boardgame, he/she probably liked the game to begin with. Light users tend to buy games that they have played elsewhere and enjoyed already instead of buying new games they haven't yet played.

To my advanced user, I wanted to recommend advanced games so I used a Pearson similarity. Pearson similarity is essentially the same as cosine similarity adjusted for the mean rating of each game. This worked really well in accentuating the subtle differences from the game's mean rating. Advanced users try out new games, some they like, others they don't. They also rate stringently. So utilizing the subtle deviations from the mean rating makes sense for this userbase. The result is a more complex set of recommendations, which is exactly how I want it to be.

I also made a Funk-SVD model (Netflix 1MM prize winner) for my advanced users with game attributes as side data (playing time, # players, suggestedd age etc). This model works on user-user + item-item similarites and with the additional features, provides a much more specific recommendation. However, for starting off, It makes sense to use the Pearson model instead of the Funk-SVD in order to get a real world baseline. Once deployed, I would validate with precision not F1 since we can't ask users to rate all 5000 games and thus cannot get an accurate recall. After validating the Pearson, Jaccard and Cosine models, I would employ the Funk-SVD model. 
