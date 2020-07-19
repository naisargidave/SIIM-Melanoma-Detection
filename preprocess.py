# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 21:02:55 2020

@author: naisa
"""

import pandas as pd
import numpy as np

data = pd.read_csv('C:\\Users\\naisa\\Desktop\\SIIM-ISIC Melanoma\\train.csv')
test_data = pd.read_csv('C:\\Users\\naisa\\Desktop\\SIIM-ISIC Melanoma\\test.csv')

data = data.drop(columns = ['patient_id','diagnosis','benign_malignant'])
test_data = test_data.drop(columns = ['patient_id'])

label_freq = data['anatom_site_general_challenge'].value_counts().reset_index()    
labels_anatom = np.array(label_freq['index'])
freqs_anatom = np.array(label_freq['anatom_site_general_challenge'])
freqs_anatom = freqs_anatom / np.sum(freqs_anatom)

label_freq = data['age_approx'].value_counts().reset_index()    
labels_age = np.array(label_freq['index'])
freqs_age = np.array(label_freq['age_approx'])
freqs_age = freqs_age / np.sum(freqs_age)

label_freq = data['sex'].value_counts().reset_index()    
labels_sex = np.array(label_freq['index'])
freqs_sex = np.array(label_freq['sex'])
freqs_sex = freqs_sex / np.sum(freqs_sex)



def fill(value):
    value1 = value['anatom_site_general_challenge']

    value1 = str(value1)
    if value1 == "nan":
        value['anatom_site_general_challenge'] = np.random.choice(labels_anatom, size = 1, replace = False, p = freqs_anatom)[0]
    else:
        value['anatom_site_general_challenge'] = value1
        
    value1 = value['age_approx']

    if np.isnan(value1):
        value['age_approx'] = np.random.choice(labels_age, size = 1, replace = False, p = freqs_age)[0]
    else:
        value['age_approx'] = value1
        
    value1 = value['sex']

    value1 = str(value1)
    if value1 == "nan":
        value['sex'] = np.random.choice(labels_sex, size = 1, replace = False, p = freqs_sex)[0]
    else:
        value['sex'] = value1
        
    
        
    return value

data = data.apply(fill, axis = 1)
data['train_test'] = 'train'
test_data = test_data.apply(fill, axis=1)
test_data['train_test'] = 'test'
combined_data = pd.concat([data,test_data])
one_hot_sex = pd.get_dummies(combined_data['sex'], prefix = 'sex')
combined_data = pd.concat([combined_data, one_hot_sex], axis = 1)
one_hot_anatom = pd.get_dummies(combined_data['anatom_site_general_challenge'], prefix = 'anatom')
combined_data = pd.concat([combined_data, one_hot_anatom], axis = 1)
combined_data = combined_data.drop(columns = ['sex','anatom_site_general_challenge'])
combined_data['age_approx'] = (combined_data['age_approx']-combined_data['age_approx'].min())/(combined_data['age_approx'].max()-combined_data['age_approx'].min())

train_data = combined_data[combined_data['train_test'] == 'train']
train_data = train_data.drop(columns = ['train_test'])
test_data = combined_data[combined_data['train_test'] == 'test']
test_data = test_data.drop(columns = ['train_test','target'])

train_data.to_csv('C:\\Users\\naisa\\Desktop\\SIIM-ISIC Melanoma\\final_train.csv', index = False)
test_data.to_csv('C:\\Users\\naisa\\Desktop\\SIIM-ISIC Melanoma\\final_test.csv', index = False)




#
#data_subset_1 = data[data['target'] == 1]
#
#data_subset_0 = data[data['target'] == 0].sample(len(data_subset_1)) 
#
#data_subset = pd.concat([data_subset_1,data_subset_0])
#
#one_hot_sex = pd.get_dummies(data_subset['sex'], prefix = 'sex')
#
#data_subset = data_subset.join(one_hot_sex)
#
#one_hot_anatom = pd.get_dummies(data_subset['anatom_site_general_challenge'], prefix = 'anatom')
#
#data_subset = data_subset.join(one_hot_anatom)

#data_subset = data_subset.drop(columns = ['sex','anatom_site_general_challenge'])


#x = data_subset.iloc[17]['target']
#
#final_columns = [x for x in data_subset.columns if 'sex' in x or 'anatom' in x or 'approx' in x]
#
#data_subset['age_approx'] = (data_subset['age_approx']-data_subset['age_approx'].min())/(data_subset['age_approx'].max()-data_subset['age_approx'].min())

#
#labels = data_subset['age_approx'][pd.notnull(data_subset['age_approx'])].unique()
#
#bins = len(labels)
#
#val, bins = np.histogram(data_subset['age_approx'][data_subset['age_approx'].notnull()], bins=bins, density = False)
#


