import json
from math import exp

import numpy as np


def get_entries(file_path, flag):
    dict = {}
    with open(file_path) as datafile:
        for line in datafile:
            dict[line.strip()] = flag
    return dict


def get_cluster(file):
    lookup_dict = {}
    aspect_index = {}
    index = 0
    with open(file) as datafile:
        for line in datafile:
            aspect = line.split(":")[0].strip()
            aspect_index[aspect] = index
            index += 1
            features = line.split(":")[1].strip().split(",")
            for feature in features:
                feature = feature.strip();
                lookup_dict[feature] = aspect
    return [lookup_dict, aspect_index]


def get_reviews(file_path):
    user_dict = {}
    product_dict = {}
    with open(file_path) as datafile:
        for line in datafile:
            json_data = json.loads(line)
            user_id = json_data["userID"]
            product_id = json_data["productID"]
            overall = json_data["overall"]
            feature = json_data["feature"]
            opinion = json_data["opinion"]
            if user_id not in user_dict:
                user_dict[user_id] = []
            if product_id not in product_dict:
                product_dict[product_id] = []
            user_dict[user_id].append([feature, opinion])
            product_dict[product_id].append([feature, opinion])
    return [user_dict, product_dict]


def get_index(user_dict, product_dict):
    user_index = {}
    product_index = {}
    index = 0
    for user in user_dict.keys():
        user_index[user] = index
        index += 1
    index = 0
    for product in product_dict.keys():
        product_index[product] = index
        index += 1
    return [user_index, product_index]


def get_user_item_matrix(file_path, user_index, product_index):
    num_users = len(user_index)
    num_product = len(product_index)
    result = np.zeros((num_users, num_product))
    with open(file_path) as datafile:
        for line in datafile:
            json_data = json.loads(line)
            user_id = json_data["userID"]
            product_id = json_data["productID"]
            user = user_index[user_id]
            product = product_index[product_id]
            overall = json_data["overall"]
            result[user, product] = overall
    return result


def get_user_feature_matrix(user_dict, user_index, lookup_dict, aspect_index, N):
    result = np.zeros((len(user_index), len(aspect_index)))
    for key in user_dict.keys():
        index_user = user_index[key]
        user_reviews = user_dict[key]
        count_dict = {}
        for review in user_reviews:
            feature = review[0]
            if feature not in lookup_dict:
                continue
            aspect = lookup_dict[feature]
            if aspect not in count_dict:
                count_dict[aspect] = 0;
            count_dict[aspect] += 1
        for aspect in count_dict.keys():
            index_aspect = aspect_index[aspect]
            count = count_dict[aspect]
            result[index_user, index_aspect] = 1 + (N - 1) * (2 / (1 + exp(-count)) - 1)
    return result


def get_product_feature_matrix(product_dict, product_index, lookup_dict, aspect_index, N, neg_dict, pos_dict):
    result = np.zeros((len(product_index), len(aspect_index)))
    for key in product_dict.keys():
        index_product = product_index[key]
        product_reviews = product_dict[key]
        count_dict = {}
        for review in product_reviews:
            reverse = False
            feature = review[0]
            opinion = review[1]
            neg_set = ["not", "n't"]
            if feature not in lookup_dict:
                continue
            if len(opinion.split()) > 1 and opinion.split()[0] in neg_set:
                reverse = True
                opinion = opinion.split()[1]
            if opinion in neg_dict:
                s = -1
            elif opinion in pos_dict:
                s = 1
            else:
                continue
            if reverse:
                s = -s
            aspect = lookup_dict[feature]
            if aspect not in count_dict:
                count_dict[aspect] = [];
            count_dict[aspect].append(s)
        for aspect in count_dict.keys():
            index_aspect = aspect_index[aspect]
            count = sum(count_dict[aspect])
            result[index_product, index_aspect] = 1 + (N - 1) / (1 + exp(-count))
    return result


if __name__ == "__main__":
    feature_cluster_path = "../data/Cell_Phones_and_Accessories_5/feature-cluster.txt"
    reviews_path = "../data/Cell_Phones_and_Accessories_5/extractedReviews.txt"
    neg_entries = "../data/Cell_Phones_and_Accessories_5/opinion-lexicon-English/negative-words.txt"
    pos_entries = "../data/Cell_Phones_and_Accessories_5/opinion-lexicon-English/positive-words.txt"
    [lookup_dict, aspect_index] = get_cluster(feature_cluster_path)
    [user_dict, product_dict] = get_reviews(reviews_path)
    [user_index, product_index] = get_index(user_dict, product_dict)
    user_item_matrix = get_user_item_matrix(reviews_path, user_index, product_index)
    user_feature_matrix = get_user_feature_matrix(user_dict, user_index, lookup_dict, aspect_index, 5)
    neg_dict = get_entries(neg_entries, -1)
    pos_dict = get_entries(pos_entries, 1)
    product_feature_matrix = get_product_feature_matrix(product_dict, product_index, lookup_dict, aspect_index, 5,
                                                        neg_dict, pos_dict)
