import get_matrices
import train
import numpy as np

def get_user_care(user_id, user_index, aspect_index, user_feature_matrix):
    user_i = user_index[user_id]
    for k, v in aspect_index.items():
        print k + " : " + str(user_feature_matrix[user_i, v])

def get_item_care(product_id, product_index, aspect_index, product_feature_matrix):
    user_i = product_index[product_id]
    for k, v in aspect_index.items():
        print k + " : " + str(product_feature_matrix[user_i, v])

def top_k(product_index, user_id, X, Y, A, user_index, user_feature_matrix, k, alpha):
    user_i = user_index[user_id]
    user_care = user_feature_matrix[user_i, :]
    idx = np.argpartition(user_care, -k)
    idx = idx[-k:]
    R_i = np.zeros(Y.shape[0])
    for i in range(R_i.shape[0]):
        tmp = X[user_i, idx].dot(Y[i, idx].T) / k / 5.0
        R_i[i] = tmp * alpha + (1 - alpha) * A[user_i, i]
    idx = np.argpartition(R_i, -3)
    idx = idx[-3:]
    item_id = get_item_id(product_index, idx[-1])
    return item_id

def get_item_id(product_index, index):
    for k, v in product_index.items():
        if v == index:
            return k



if __name__ == "__main__":
    feature_cluster_path = "../data/Cell_Phones_and_Accessories_5/feature-cluster.txt"
    reviews_path = "../data/Cell_Phones_and_Accessories_5/extractedReviews.txt"
    neg_entries = "../data/Cell_Phones_and_Accessories_5/opinion-lexicon-English/negative-words.txt"
    pos_entries = "../data/Cell_Phones_and_Accessories_5/opinion-lexicon-English/positive-words.txt"
    [lookup_dict, aspect_index] = get_matrices.get_cluster(feature_cluster_path)
    [user_dict, product_dict] = get_matrices.get_reviews(reviews_path)
    [user_index, product_index] = get_matrices.get_index(user_dict, product_dict)
    user_item_matrix = get_matrices.get_user_item_matrix(reviews_path, user_index, product_index)
    user_feature_matrix = get_matrices.get_user_feature_matrix(user_dict, user_index, lookup_dict, aspect_index, 5)
    neg_dict = get_matrices.get_entries(neg_entries, -1)
    pos_dict = get_matrices.get_entries(pos_entries, 1)
    product_feature_matrix = get_matrices.get_product_feature_matrix(product_dict, product_index, lookup_dict, aspect_index, 5,
                                                        neg_dict, pos_dict)
    u_id = user_index["AYB4ELCS5AM8P"]
    p_id = product_index["B00HH7MAUW"]
    user_item_matrix[u_id, p_id] = 0
    [U1, U2, V, H1, H2] = train.training(user_item_matrix, user_feature_matrix, product_feature_matrix, 50, 50, 0.01, 0.01, 0.01, 0.01, 0.01, 5000, 0.002)
    X_ = U1.dot(V.T)
    Y_ = U2.dot(V.T)
    A_ = U1.dot(U2.T) + H1.dot(H2.T)
    print A_[u_id, p_id]
    #most_rec = top_k(product_index, "AYB4ELCS5AM8P", X_, Y_, A_, user_index, user_feature_matrix, 5, 1.0)
    #print most_rec
    #print get_user_care("A10UHQH1YL5Q6B", user_index, aspect_index, user_feature_matrix)
    #print get_item_care(most_rec, product_index, aspect_index, product_feature_matrix)