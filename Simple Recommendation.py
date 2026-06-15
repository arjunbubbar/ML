# simple recommendor, content based - personal interests, collaborative filtering - similarity in consumers

import pandas as pd

moviesdf = pd.read_csv ('/Users/arjunbubbar/Desktop/Jetlearn/Data Science/Datasets/movies_metadata.csv')

print (moviesdf.info())

# weighted rating (v/(v+m))*R + (m/(v+m))*C 
# v = number of votes - vote count
# m = minimum number of votes required to be listed
# R = average rating for movie - vote average
# C = mean vote across whole data - average of vote average

v = moviesdf ['vote_count']
m = moviesdf ['vote_count'].quantile (0.90)
r = moviesdf ['vote_average']
c = moviesdf ['vote_average'].mean ()

moviesdf ['weightedrating'] = (v/(v+m))*r + (m/(v+m))*c

moviesdf = moviesdf.sort_values ('weightedrating', ascending=False)

print (moviesdf ['title'].head (20))