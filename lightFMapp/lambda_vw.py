import pandas as pd
import json
import numpy as np
import sys
import os
import boto3
import botocore
from boto3.dynamodb.conditions import Key, Attr
from datetime import datetime
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from lightfm import LightFM



# number_of_actions = 10


# currently deployed static ids which have overall highest score in the DB,  will be dynamic products ids based on client IDs

def load_model(path_to_model):
    # load model saved as pickel file
    pass

def lambda_handler(event, context):
    user_activity = {
        "client_id": event['client_id'],
        "products_clicked": event['product_clicked'],
        "cart_details": event['cart_detail'],
        "products_purchased": event['product_purchased'],
        "session": event['status'],
    }
    dynamodb = boto3.client('dynamodb')
    response = dynamodb.scan(TableName='product_db_rec_rl', Select='ALL_ATTRIBUTES')
    data = response['Items']
    while 'LastEvaluatedKey' in response:
        response = dynamodb.scan(TableName='product_db_rec_rl', Select='ALL_ATTRIBUTES',
                                 ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])
    df = pd.DataFrame()
    # product_id = []
    # product_name = []
    # product_CTR = []
    # product_add_to_carts = []
    # product_brand = []
    # product_category = []
    # product_color = []
    # product_price = []
    # product_units_sold = []
    for i in range(len(data)):
        product_id.append(int(data[i]['product_id']['N']))
        product_name.append(str(data[i]['product_name']['S']))
        product_CTR.append(int(data[i]['product_CTR']['N']))
        product_add_to_carts.append(int(data[i]['product_add_to_carts']['N']))
        product_brand.append(str(data[i]['product_brand']['S']))
        product_category.append(str(data[i]['product_category']['S']))
        product_color.append(str(data[i]['product_color']['S']))
        product_price.append(int(data[i]['product_price']['N']))
        product_units_sold.append(int(data[i]['product_units_sold']['N']))
    df['product_id'] = product_id
    df['product_name'] = product_name
    df['product_CTR'] = product_CTR
    df['product_add_to_carts'] = product_add_to_carts
    df['product_brand'] = product_brand
    df['product_category'] = product_category
    df['product_color'] = product_color
    df['product_price'] = product_price
    df['product_units_sold'] = product_units_sold
    df_jeans = df[df.product_category == 'jeans'].reset_index().drop(['index'], axis=1)
    print(df_jeans.columns)
    if event['model'] == 'rl':
        s3 = boto3.resource('s3')
        key_name = 'recommendation.model'
        file_name = '/tmp/recommendation.model'
        bucket_name = 'old-customer-data-bucket'

        try:
            s3.Bucket(bucket_name).download_file(key_name, file_name)
        except:
            return 'Model Load Failed.'

        vw = load_model('/tmp/recommendation.model')
        # train_feature = event
        # _product_db = df

        # for index,row in _product_db.iterrows():
        # if row["product_id"] in _product_clicked:
        #     feature1.append(str(row))
        # if row["product_id"] in _product_purchased:
        #     feature2.append(str(row))
        # if row["product_id"] in _add_to_carts:
        #     feature3.append(str(row))

        # feature1 = train_feature["client_id"]
        # feature2 = train_feature["product_clicked"]
        # feature3 = train_feature["cart_detail"]
        # feature4 = train_feature["status"]#current status is only mid session and end session, to be replaced with stages of the
        # feature5 = train_feature["product_purchased"]

        # for index,row in _product_db.iterrows():
        # if row["product_id"] in _product_clicked:
        #     feature1.append(str(row))
        # if row["product_id"] in _product_purchased:
        #     feature2.append(str(row))
        # if row["product_id"] in _add_to_carts:
        #     feature3.append(str(row))

        # model_input = "| " + str(feature1) + " " + str(feature2) + " " + str(feature3) + " " + str(feature4) + " " + str(feature5)
        _product_clicked = user_activity['products_clicked']
        _add_to_cart = user_activity['cart_details']
        _product_purchased = user_activity['products_purchased']
        feature1 = []
        feature2 = []
        feature3 = []
        feature4 = user_activity['client_id']
        feature5 = user_activity['session']
        for index, row in df_jeans.iterrows():
            if index in _product_clicked:
                feature1.append(str(row))
            if index in _add_to_cart:
                feature2.append(str(row))
            if index in _product_purchased:
                feature3.append(str(row))
        model_input = "| " + str(feature1) + " " + str(feature2) + " " + str(feature3) + " " + str(
            feature4) + " " + str(feature5)
        result = vw.predict(model_input)
        action = np.argmax(result)
        probability = max(result)
        products = {}

        # max_sellers = [32,34,545,65,7676,7342,34,5,23,98]
        max_sellers = [1, 2, 3, 10, 20, 7, 8, 6, 20, 30]
        tuple_product_scores = []
        for i in existing_product_scores.keys():
            tuple_b = (existing_product_scores[i], i)
            tuple_product_scores.append(tuple_b)

        sorted_tuples = sorted(tuple_product_scores, reverse=True)
        for i in range(10):
            try:
                max_seller[i] = sorted_tuple[i][1]
            except:
                max_seller[i] = 0

        for i in range(5):
            k = np.argmax(result)
            products[max_sellers[k]] = result[k]
            del result[k]

        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('user_json_data')

        try:
            response_dynamodb = table.query(KeyConditionExpression=Key('client_id').eq(event['client_id']))
        except:
            return "Error in fetching data from DB."

        existing_product_scores = {}

        for j in range(len(response_dynamodb['Items'])):
            product_name = response_dynamodb['Items'][j]['product_id']
            existing_product_scores[product_name] = response_dynamodb['Items'][j]['score']

        for j in products.keys():
            if j in existing_product_scores.keys():
                old_score = existing_product_scores[j]
                new_score = products[j] * 100
                score_to_update = int((old_score + new_score) / 2)
                existing_product_scores[j] = score_to_update
            else:
                existing_product_scores[j] = products[j] * 100

        for j in existing_product_scores.keys():
            try:
                table.put_item(
                    Item={
                        'client_id': str(event['client_id']),
                        'product_id': str(j),
                        'score': int(existing_product_scores[j])
                    })
            except botocore.exceptions.ClientError as e:
                return e
        table_action_probability = dynamodb.Table('action_probablitiy_database')

        try:
            table_action_probability.put_item(
                Item={
                    'client_id': str(event['client_id']),
                    'action': int(action),
                    'probability': int(probability * 100)
                })
        except botocore.exceptions.ClientError as e:
            return e

        return "Predictions Saved to DB succesfully."
    else:
        number_of_actions = 10
        bucket_name = 'old-customer-data-bucket'
        s3 = boto3.resource('s3')
        # try:
        #     s3.Bucket(bucket_name).download_file('jeans.model' , '/tmp/jeans.model')
        #     vw = load_model(path_to_model = '/tmp/jeans.model' , number_of_actions = number_of_actions)
        # except botocore.exceptions.ClientError as e:
        #     print (e)
        #     print ("Creating New Model")
        #     vw = pyvw.vw("--cb_explore " + str(number_of_actions))
        _product_clicked = user_activity['products_clicked']
        _add_to_cart = user_activity['cart_details']
        _product_purchased = user_activity['products_purchased']
        feature1 = []
        feature2 = []
        feature3 = []
        feature4 = user_activity['client_id']
        feature5 = user_activity['session']
        for index, row in df_jeans.iterrows():
            if index in _product_clicked:
                feature1.append(str(row))
            if index in _add_to_cart:
                feature2.append(str(row))
            if index in _product_purchased:
                feature3.append(str(row))
        model_input = "| " + str(feature1) + " " + str(feature2) + " " + str(feature3) + " " + str(
            feature4) + " " + str(feature5)
        s3 = boto3.resource('s3')
        bucket_name = 'old-customer-data-bucket'
        key_name = 'recommendation.model'
        file_name = '/tmp/recommendation.model'
        try:
            s3.Bucket(bucket_name).download_file(key_name, '/tmp/recommendation.model')
            flag = True
        except:
            flag = False

        if flag == True:
            vw = load_model('/tmp/recommendation.model')
        else:
            vw = pyvw.vw("--cb_explore " + str(number_of_actions))

        # Prediction
        # train_feature = event
        # feature1 = train_feature["client_id"]
        # feature2 = train_feature["product_clicked"]
        # feature3 = train_feature["cart_detail"]
        # feature4 = train_feature["status"]
        # feature5 = train_feature["product_purchased"]
        # model_input = "| " + str(feature1) + " " + str(feature2) + " " + str(feature3) + " " + str(feature4) + " " + str(feature5)
        result = vw.predict(model_input)
        action = np.argmax(result)
        probability = max(result)
        products = {}
        max_sellers = [1, 2, 3, 10, 20, 7, 8, 6, 20, 30]
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table('user_json_data')
        try:
            response_dynamodb = table.query(KeyConditionExpression=Key('client_id').eq(event['client_id']))
        except:
            return "Error in fetching data from DB."
        existing_product_scores = {}

        for j in range(len(response_dynamodb['Items'])):
            product_name = response_dynamodb['Items'][j]['product_id']
            existing_product_scores[product_name] = response_dynamodb['Items'][j]['score']
        tuple_product_scores = []
        for i in existing_product_scores.keys():
            tuple_b = (existing_product_scores[i], i)
            tuple_product_scores.append(tuple_b)

        sorted_tuples = sorted(tuple_product_scores, reverse=True)
        for i in range(10):
            try:
                max_sellers[i] = sorted_tuple[i][1]
            except:
                max_sellers[i] = 0

        for i in range(5):
            k = np.argmax(result)
            products[max_sellers[k]] = result[k]
            del result[k]
        dynamodb = boto3.resource('dynamodb')

        for j in products.keys():
            if j in existing_product_scores.keys():
                old_score = existing_product_scores[j]
                new_score = products[j] * 100
                score_to_update = int(((old_score) + (new_score)) / 2)
                existing_product_scores[j] = score_to_update
            else:
                existing_product_scores[j] = products[j] * 100

        for j in existing_product_scores.keys():
            try:
                table.put_item(
                    Item={
                        'client_id': str(event['client_id']),
                        'product_id': str(j),
                        'score': int(existing_product_scores[j])
                    })
            except botocore.exceptions.ClientError as e:
                return e
        table_action_probability = dynamodb.Table('action_probablitiy_database')

        try:
            response_dynamodb = table_action_probability.query(
                KeyConditionExpression=Key('client_id').eq(str(event['client_id'])))
        except botocore.exceptions.ClientError as e:
            return e

        # try:
        #     table_action_probability.put_item(
        #         Item = {
        #             'client_id' : str(event['client_id']) ,
        #             'action' : int(action) ,
        #             'probability' : int(probability)
        #         })
        # except botocore.exceptions.ClientError as e:
        #     return e
        # atm_data_key_name = 'activity_tracker_module_data.csv'
        # chat_features_key_name = 'chat_feature_module_data.csv'
        # bucket_name = 'old-customer-data-bucket'
        # try:
        #     s3.Bucket(bucket_name).download_file(atm_data_key_name , '/tmp/atm.csv')
        # except botocore.exceptions.ClientError as e:
        #     print ("Activity Tracker Module Data Download from S3 Failed.")
        #     return e
        # try:
        #     s3.Bucket(bucket_name).download_file(chat_features_key_name , '/tmp/chat.csv')
        # except botocore.exceptions.ClientError as e:
        #     print ("Chat Features Data Download from S3 Failed.")
        #     return
        # df = pd.read_json(event)
        train_feature = event

        cost = 0

        max_sellers = [32, 34, 545, 65, 7676, 7342, 34, 5, 23, 98]

        if len(response_dynamodb['Items']) > 0:
            action = int(response_dynamodb['Items'][0]['action'])
            # action = float(response_dynamodb['Items'][0]['action']/100)
            probability = float(response_dynamodb['Items'][0]['probability'] / 100)
        else:
            action = 0
            probability = 0

        if (max_sellers[action] in train_feature["product_clicked"]):
            cost = -10
        elif (max_sellers[action] in train_feature["product_purchased"]):
            cost = -20
        elif (max_sellers[action] in train_feature["cart_detail"]):
            cost = -30
        else:
            cost = 0

        feature1 = train_feature["client_id"]
        feature2 = train_feature["product_clicked"]
        feature3 = train_feature["cart_detail"]
        feature4 = train_feature["status"]
        feature5 = train_feature["product_purchased"]

        # Construct the example in the required vw format

        # Construct the example in the required vw format.
        learn_example = str(action) + ":" + str(cost) + ":" + str(probability) + " | " + str(feature1) + " " + str(
            feature2) + " " + str(feature3) + " " + str(feature4) + " " + str(feature5)

        # Here we do the actual learning.
        # Save Action & Probability to DB : action_cost_database
        action_cost_table = dynamodb.Table('action_cost_database')
        try:
            action_cost_table.put_item(
                Item={
                    'client_id': str(user_activity['client_id']),
                    'timestamp': str(datetime.now()),
                    'action': int(action),
                    'probability': int(probability * 100),
                    'cost': cost
                })
        except botocore.exceptions.ClientError as e:
            return e
        vw.learn(learn_example)
        vw.save('/tmp/recommendation.model')
        try:
            response = s3.Bucket(bucket_name).upload_file('/tmp/recommendation.model', 'recommendation.model')
            return "Model Saved Successfully"
        except botocore.exceptions.ClientError as e:
            return e