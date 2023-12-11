#######################################################################################################
Q1) Build a recommender system with the given data using UBCF.

This dataset is related to the video gaming industry and a survey was conducted to build a 
recommendation engine so that the store can improve the sales of its gaming DVDs. Snapshot of the dataset is given below. Build a Recommendation Engine and suggest top selling DVDs to the store customers.

##########################################################################################################

import pandas as pd

game = pd.read_csv("D:/Hands on/13_Network Analysis and Recommandation sys/Assignment/game.csv")

game.isnull().sum()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words = 'english')

tfidf_matrix = tfidf.fit_transform(game['game'])

tfidf_matrix.shape

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

game_index = pd.Series(game.index, index = game.game).drop_duplicates()

game_id = game_index['The Legend of Zelda: Ocarina of Time']
#game_id = game_index['Grand Theft Auto IV']

game_id

def get_recommandations(name, topN):
    
    game_id = game_index[name]
    
    cosine_score = list(enumerate(cosine_sim_matrix[game_id]))
    
    cosine_score = sorted(cosine_score, key = lambda x:x[1], reverse = True)
    
    cosine_scoreN = cosine_score[0 : topN + 1]
    
    game_idx = [i[0] for i in cosine_scoreN]
    game_score = [i[1] for i in cosine_scoreN]
    
    game_sim_show = pd.DataFrame(columns = ['game', 'score'])
    game_sim_show['game'] = game.loc[game_idx, 'game']
    game_sim_show['score'] = game_score
    game_sim_show.reset_index(inplace = True)
    return game_sim_show.iloc[1:, ]

rec = get_recommandations('The Legend of Zelda: Ocarina of Time', 5)
rec
   
    
##########################################################################################################

Q2) The Entertainment Company, which is an online movie watching platform, wants to improve its collection 
    of movies and showcase those that are highly rated and recommend those movies to its customer by their
    movie watching footprint. For this, the company has collected the data and shared it with you to 
    provide some analytical insights and also to come up with a recommendation algorithm so that it can 
    automate its process for effective recommendations. The ratings are between -9 and +9.
    
###########################################################################################################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("D:/Hands on/13_Network Analysis and Recommandation sys/Assignment/Entertainment.csv")

tfidf = TfidfVectorizer(stop_words = 'english')

tfidf_matrix = tfidf.fit_transform(data.Category)

tfidf_matrix.shape

cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

data_indx = pd.Series(data.index, index = data.Titles).drop_duplicates()

data_id = data_indx['Heat (1995)']

def recommendation(name, topN):
    
    data_id = data_indx[name]
    
    cosine_score = list(enumerate(cosine_sim_matrix[data_id]))
    
    cosine_score = sorted(cosine_score, key = lambda x:x[1], reverse = True)
    
    cosine_scoreN = cosine_score[0 : topN + 1]
    
    game_index = [i[0] for i in cosine_scoreN]
    game_score = [i[1] for i in cosine_scoreN]
    
    game_sim_show = pd.DataFrame(columns = ['names', 'score'])
    game_sim_show['names'] = data.loc[game_index, 'Titles']
    game_sim_show['score'] = game_score
    game_sim_show.reset_index(inplace = True)
    return game_sim_show.iloc[1:, ]


rec = recommendation('Babe (1995)', 5)
rec
    