import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')

credits_renamed = credits.rename(index=str,columns={"movie_id":"id"})
movie_dirty = movies.merge(credits_renamed,on='id')

movies_clean = movie_dirty.drop(columns=['homepage','title_x','title_y','status','production_countries'])

## Get highest weighted(most votes and ratings)

V = movies_clean['vote_count']
R = movies_clean['vote_average']
C = movies_clean['vote_average'].mean()
m = movies_clean['vote_count'].quantile(0.70)

movies_clean['weighted_average'] = (V/(V+m) * R) + (m/(m+V) * C)

movies_ranked = movies_clean.sort_values('weighted_average',ascending=False)

plt.figure(figsize=(10,10))

ax = sns.barplot(x=movies_ranked['weighted_average'].head(10),y=movies_ranked['original_title'].head(10))
plt.xlim(6.75,8.35)
plt.title('"Best" Movies by TMDB Votes', weight='bold')
plt.xlabel('Weighted Average Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')


## Get most popular movies

popular = movies_clean.sort_values('popularity',ascending=False)

plt.figure(figsize=(10,10))
ax = sns.barplot(x=popular['popularity'].head(10), y=popular['original_title'].head(10), data=popular, palette='deep')

plt.title('"Most Popular" Movies by TMDB Votes', weight='bold')
plt.xlabel('Popularity Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')


## Combination of highest rated and popular

from sklearn import preprocessing

min_max = preprocessing.MinMaxScaler()
movies_scaled = min_max.fit_transform(movies_clean[['popularity','weighted_average']])
movies_norm = pd.DataFrame(movies_scaled,columns=['popularity','weighted_average'])

movies_clean[['norm_weighted_average', 'norm_popularity']] = movies_norm

movies_clean['score'] = movies_clean['norm_weighted_average'] * 0.5 + movies_clean['norm_popularity'] * 0.5

movies_scored = movies_clean.sort_values('score',ascending=False)

plt.figure(figsize=(10,10))
ax = sns.barplot(movies_scored['score'].head(10),movies_scored['original_title'].head(10))
plt.title('Best Rated & Most Popular Blend', weight='bold')
plt.xlabel('Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')

## Content based filtering

from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer(min_df=3, max_features=None,
                      strip_accents='unicode', analyzer='word',
                      token_pattern=r'\w{1,}',ngram_range=(1,3),
                      use_idf=1,smooth_idf=1, sublinear_tf=1,
                      stop_words='english')


movies_clean['overview'] = movies_clean['overview'].fillna('')

tfv_matrix = tfv.fit_transform(movies_clean['overview'])

from sklearn.metrics.pairwise import  sigmoid_kernel

#Signoid kernal
sig = sigmoid_kernel(tfv_matrix,tfv_matrix)

indices = pd.Series(movies_clean.index,index=movies_clean['original_title']).drop_duplicates()

def give_rec(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices[title]

    # Get the pairwsie similarity scores
    sig_scores = list(enumerate(sig[idx]))

    # Sort the movies
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]

    # Movie indices
    movie_indices = [i[0] for i in sig_scores]

    # Top 10 most similar movies
    return movies_clean['original_title'].iloc[movie_indices]

give_rec('Spy Kids')




list = [1,2,3,4,5]


