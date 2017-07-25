I love boardgames and I made a boardgame recommender for casual and advanced users. It involved scraping Boardgamegeek.com for 0.5 billion ratings across 5000 boardgames for 165,000 users. 

The data indicates that advanced users rate games more critically than casual users. This makes sense and is evident in the rating variances of the two subsets.

To my casual user, I'd like to recommend more popular boardgames like Catan, Carcasonne and Banagrams but to my advanced user, I want a much more personalized recommendation. So I used Pearson similarity to estimate ratings for advanced users, which is essentially the same as a cosine similarity adjusted for the mean of each game. This worked really well in accentuating the minor differences from the mean, which is quite meaningful for advanced users. 

For casual users, I used a Jaccard and Cosine similarity metric, both of which performed similarly in RMSE, which means that the fact that a casual user has rated a particular game is much more important than the value of the rating itself! So my final model for this user subset uses Jaccard similarity. 

There is much more to be done on this dataset including joining it with game data, which is readily available. Using features like suggested playing time, mechanics (dice roll, cards, strategy etc.), I'd like to further personalize the recommendations. 

I have already tried a funk-SVD model for my advanced users with this type of data as side features. However, for the sake of the first recommender, since none exist in the market already, a simple approach like jaccard & pearson would be a smart first pass. After validating with these models using RMSE and even precision/recall as a metric, I would then try the funk SVD model with engineered features.

Feel free to connect with me if you'd like to chat more about this project.

Regards,
Aamir
