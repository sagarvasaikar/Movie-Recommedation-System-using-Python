# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:02:47 2020

@author: Sagar
"""

#import libraries needed
import pandas as pd 
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_index_from_title(title):
     return movie[movie.title== title]["index"].values[0]
 
def get_title_from_index(index):
     return movie[movie.index== index]["title"].values[0]
 
    
#import dataset
movie= pd.read_csv("C:/Users/User/Desktop/Movie Recommender/movie_dataset.csv")
movie.head()
movie.shape
movie.columns
movie.title
#remove NaN values
features=["genres", "keywords", "cast", "director"]
for feature in features :
    movie[feature]=movie[feature].fillna(" ")

#features selection
features=["genres", "keywords", "cast", "director"]

#combine all features in a dataframe
def combined_features(row):
    return row['keywords']+" "+ row['cast']+" "+ row['genres']+" "+ row['director']

movie["combined_features"] =movie.apply(combined_features,axis=1)

#count matrix
cv= CountVectorizer()
count_matrix= cv.fit_transform(movie["combined_features"])

#cosine similiarity
cosine_sim =cosine_similarity(count_matrix)
user_movie= "The Godfather"

#get index of movie

movie_index = get_index_from_title(user_movie)
movie_index

similar_movies = list(enumerate(cosine_sim[movie_index]))
similar_movies

#sort similar movies
sort_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)

#print top 10 similar movies 

i=0
for a in sort_similar_movies :
    print (get_title_from_index(a[0]))
    i=i+1
    if i>10 :
        break

result=[]
i=0
for a in sort_similar_movies :
    c= get_title_from_index(a[0])
    result.append(c)
    i=i+1
    if i>10 :
        break

result

#save model to disk
pickle.dump(result,open('model.pkl','wb'))

#load model to see results

model=pickle.load(open('model.pkl','rb'))
model

