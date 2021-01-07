import sys
import os
# sys.path.insert(1, '/Users/ajayguru/Desktop/Work/PEYE/recommend-testing/cookbook')
# sys.path.append(os.path.abspath("../"))
# from recsys import *
# from generic_preprocessing import *
import pandas as pd
import numpy as np
from IPython.display import HTML
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from lightfm import LightFM
import pickle

def lambda_handler(event,context):

    def train():
        ratings = pd.readcsv("ratings.csv")
        print("ratings DB loaded sucessfully")
        def create_interaction_matrix(df, user_col, item_col, rating_col, norm=False, threshold=None):
            '''
            Function to create an interaction matrix dataframe from transactional type interactions
            Required Input -
                - df = Pandas DataFrame containing user-item interactions
                - user_col = column name containing user's identifier
                - item_col = column name containing item's identifier
                - rating col = column name containing user feedback on interaction with a given item
                - norm (optional) = True if a normalization of ratings is needed
                - threshold (required if norm = True) = value above which the rating is favorable
            Expected output -
                - Pandas dataframe with user-item interactions ready to be fed in a recommendation algorithm
            '''
            interactions = df.groupby([user_col, item_col])[rating_col] \
                .sum().unstack().reset_index(). \
                fillna(0).set_index(user_col)
            if norm:
                interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
            return interactions

        interactions = create_interaction_matrix(df=ratings,
                                                 user_col='userId',
                                                 item_col='productId',
                                                 rating_col='rating',
                                                 threshold='3')

        # interactions.shape
        print("Interaction matrix created with ratings DB")

        def create_user_dict(interactions):
            '''
            Function to create a user dictionary based on their index and number in interaction dataset
            Required Input -
                interactions - dataset create by create_interaction_matrix
            Expected Output -
                user_dict - Dictionary type output containing interaction_index as key and user_id as value
            '''
            user_id = list(interactions.index)
            user_dict = {}
            counter = 0
            for i in user_id:
                user_dict[i] = counter
                counter += 1
            return user_dict

        user_dict = create_user_dict(interactions=interactions)
        print("user dictionary created")

        data_prod = pd.read_csv("product_db")
        print("product DB loaded")

        def create_item_dict(df, id_col, name_col):
            '''
            Function to create an item dictionary based on their item_id and item name
            Required Input -
                - df = Pandas dataframe with Item information
                - id_col = Column name containing unique identifier for an item
                - name_col = Column name containing name of the item
            Expected Output -
                item_dict = Dictionary type output containing item_id as key and item_name as value
            '''
            item_dict = {}
            for i in range(df.shape[0]):
                item_dict[(df.loc[i, id_col])] = df.loc[i, name_col]
            return item_dict

        products_dict = create_item_dict(df=data_prod,
                                         id_col='product_id',
                                         name_col='product_name')
        print("product dictionary created successfully")

        def runMF(interactions, n_components=30, loss='warp', k=15, epoch=30, n_jobs=4):
            '''
            Function to run matrix-factorization algorithm
            Required Input -
                - interactions = dataset create by create_interaction_matrix
                - n_components = number of embeddings you want to create to define Item and user
                - loss = loss function other options are logistic, brp
                - epoch = number of epochs to run
                - n_jobs = number of cores used for execution
            Expected Output  -
                Model - Trained model
            '''
            x = sparse.csr_matrix(interactions.values)
            model = LightFM(no_components=n_components, loss=loss, k=k)
            model.fit(x, epochs=epoch, num_threads=n_jobs)
            return model
            mf_model = runMF(interactions=interactions,
                             n_components=30,
                             loss='warp',
                             k=15,
                             epoch=30,
                             n_jobs=4)

            with open('savefile.pickle', 'wb') as fle:
                pickle.dump(model, fle, protocol=pickle.HIGHEST_PROTOCOL)


        print("Model trained sucessfully")


    def predict():
        model = pickle.load(open('savefile.pickle', 'rb'))

        def sample_recommendation_user(model, interactions, user_id, user_dict,
                                       item_dict, threshold=0, nrec_items=10, show=True):
            '''
            Function to produce user recommendations
            Required Input -
                - model = Trained matrix factorization model
                - interactions = dataset used for training the model
                - user_id = user ID for which we need to generate recommendation
                - user_dict = Dictionary type input containing interaction_index as key and user_id as value
                - item_dict = Dictionary type input containing item_id as key and item_name as value
                - threshold = value above which the rating is favorable in new interaction matrix
                - nrec_items = Number of output recommendation needed
            Expected Output -
                - Prints list of items the given user has already bought
                - Prints list of N recommended items  which user hopefully will be interested in
            '''
            n_users, n_items = interactions.shape
            user_x = user_dict[user_id]
            scores = pd.Series(model.predict(user_x, np.arange(n_items)))
            scores.index = interactions.columns
            scores = list(pd.Series(scores.sort_values(ascending=False).index))

            known_items = list(pd.Series(interactions.loc[user_id, :] \
                                             [interactions.loc[user_id, :] > threshold].index) \
                               .sort_values(ascending=False))

            scores = [x for x in scores if x not in known_items]
            return_score_list = scores[0:nrec_items]
            known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
            scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
            if show == True:
                print("Known Likes:")
                counter = 1
                for i in known_items:
                    print(str(counter) + '- ' + i)
                    counter += 1

                print("\n Recommended Items:")
                counter = 1
                for i in scores:
                    print(str(counter) + '- ' + i)
                    counter += 1
            return return_score_list

