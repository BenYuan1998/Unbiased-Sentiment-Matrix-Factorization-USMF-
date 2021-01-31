# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 22:50:31 2021

@author: Ben1998
"""
import pandas as pd
import train_test_split

file_path_d_m="../../../../../../Datasets/Amazon_datasets/Digital_Music/Amazon_digital_music_N20.txt"

file_path_p_a="../../../../../../Datasets/Amazon_datasets/Phones_Accessories/Amazon_phones_accessories_N20.txt"

file_path_f_r="../../../../../../Datasets/Yelp_datasets/Nevada_Food_Restaurant/Yelp_food_restaurants_Nevada_20core.txt"

df_d_m=train_test_split.txt_to_dataframe(file_path_d_m)

df_p_a=train_test_split.txt_to_dataframe(file_path_p_a)

df_f_r=train_test_split.txt_to_dataframe(file_path_f_r)

test_size=10

df_train_d_m=train_test_split.training_set(df_new=df_d_m,m=test_size)
df_train_d_m.to_csv("../../dat/df_train_Amazon_digital_music_N20.csv")
training_indices_d_m=df_train_d_m.index.values

df_train_p_a=train_test_split.training_set(df_new=df_p_a,m=test_size)
df_train_p_a.to_csv("../../dat/df_train_Amazon_phones_accessories_N20.csv")
training_indices_p_a=df_train_p_a.index.values

df_train_f_r=train_test_split.training_set(df_new=df_f_r,m=test_size)
df_train_f_r.to_csv("../../dat/df_train_Yelp_food_restaurants__Nevada_N20.csv")
training_indices_f_r=df_train_f_r.index.values

df_test_d_m=train_test_split.test_set(df_d_m,training_indices_d_m)
df_train_d_m.to_csv("../../dat/df_test_Amazon_digital_music_N20.csv")

df_test_p_a=train_test_split.test_set(df_p_a,training_indices_p_a)
df_test_p_a.to_csv("../../dat/df_test_Amazon_phones_accessories_N20.csv")

df_test_f_r=train_test_split.test_set(df_f_r,training_indices_f_r)
df_test_f_r.to_csv("../../dat/df_test_Yelp_food_restaurants__Nevada_N20.csv")