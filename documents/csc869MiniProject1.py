#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[3]:


"""Note: this function alters the column of variableName of the original data set 
   Function takes as input a data frame, the name of the continous variable (String), number of bins (int)
"""
def descretizer(data, variableName, binNum):

    d = data[variableName].to_numpy().reshape(len(data),1) #convert data frame to array 
    kbins = KBinsDiscretizer(n_bins=binNum, encode='ordinal', strategy='uniform') #apply kbinsDiscretizer
    data_trans = kbins.fit_transform(d) #apply kbinsDiscretizer to array
    data[variableName] = pd.DataFrame(data_trans) #convert array back to data frame and replace column in original data

    return #nothing to return since the original data column has been transformed


# In[4]:


"""function takes as input an n by k data frame and response variable (String) and returns a list of n by 2 data frames"""
def makeDataFrames(data, response):
    dataFrameList = [] #initialize empty list of data frames 
    for i in range(len(data.columns)-1): #exclude the response column (this assumes response column is the last column)
        columnName = data.columns[i]
        d = {response: data[response], columnName: data[columnName]}
        #print(pd.DataFrame(d))
        dataFrameList += [pd.DataFrame(d)] #make n by 2 data frame

    return dataFrameList


# In[5]:


"""function takes as input a list of n by 2 data frames with a categorical variable of m categories and returns a list of 2 by m dataframes of conditional probabilities"""

def makeConditionalProb(dataFramelist):
        
    conditionalProbList = [] #initialize empty list of conditional probability arrays
    
    for k in range(len(dataFramelist)): #for each predictor
        df = dataFramelist[k] #get n by 2 data frame at index k 
        columnName = df.columns[1] #get the name of the associated predictor variable 
        m = len(df[columnName].unique()) #number of predictor categories
        a = np.empty((2, m)) #initialize empty array of conditional probabilities. 2 is number of rows (high/low income). m is number of columns (number of predictor categories)
        for i in range(2):
            for j in range(m):
                category = df[columnName].unique()[j]
                x = len(df[(df.income == i) & (df[columnName] == category)]) + 1 #(for each predictor) count the total number of income i for category j 
                
                #(+ 1 prevents getting probabilities of 0)

                a[i][j] = x/(countIncome[i] + 1)

        #make array more reader-friendly
        a = pd.DataFrame(a)
        a.index = ['low income', 'high income'] #add row names
        a.columns = df[columnName].unique()  #add column names

        conditionalProbList += [a]  #add array to list of conditional probability arrays

    return conditionalProbList


# In[6]:


"""function takes in as input test data (data frame), training data (data frame), and the response label (String). Function outputs a new data set with the predicted response values 
"""
import copy 

def classifier(test_data, train_data, response):
   
    dataFrameList =  makeDataFrames(train_data, response)  #pre-process train_data into a list of n by 2 data frames
    
    conditionalProb = makeConditionalProb(dataFrameList) #store the list of conditional probability arrays
  
    predicted_data = copy.copy(test_data) #make a copy of the test_data because it will be altered 


    for i in range(len(test_data)): #for each row in test data...

        prob_low = probLowIncome #initialize prob of low income with prior
        prob_high = probHighIncome #initialize prob of high income with prior
        
        for j in range(len(test_data.loc[i][:-1])): #for each predictor in row i (len of row should always be 14); (the last value, income, is removed)

            variableName = test_data.loc[i][:-1][j] 

            #check if variableName is in train_data set 
            if (train_data.iloc[:, j] == variableName).any():

                prob_low *= conditionalProb[j][variableName][0] #multiply the conditional probability of low income given variable j 
                prob_high *= conditionalProb[j][variableName][1] #multiply the conditional probability of high income given variable j 

            else:  #if variableName is not in the train_data set, then make prob_low income/high income equal to 1 for that variableName
                prob_low *= 1
                prob_high *= 1


        if prob_high > prob_low:
            prediction = 1   
        else:
            prediction = 0

        #replace the class label with prediction 
        predicted_data.iat[i, -1] = prediction
            
    return predicted_data


# In[8]:


"""This function inputs number of k folds (int) and a data set (dataframe). The function outputs the average performance score of k crossvalidation data sets. NOTE function is very slow, but it works
"""
import copy

def kFold(k, data):
    cross_val_data = copy.copy(data) #make a copy of data 
    cross_val_data = cross_val_data.sample(frac = 1, random_state = 869).reset_index(drop = True) #randomly shuffle data 
    x = int(len(data)/k) #length of each cross validation set 
    y = {} #initialize dictionary to store the cross validation data sets

    for i in range(k):
        y[i] = cross_val_data.iloc[:x, :].reset_index(drop = True) #get the first x rows of the shuffled data
        cross_val_data = cross_val_data.iloc[x:, :] #update data set without the first x rows 

    ############################ calculate performance of cross validation #############################################

    F1_scores = []
    #accuracy_scores = []


    for i in range(k):       #for each k fold...

        train_data, test_data = kFoldHelper(i, y) #get train data and test data 

        F1_scores += [performance(train_data, test_data, "income")[0]] #get the F_1 results 
        #accuracy_scores += [performance(train_data, test_data, "income")[1]] #get the accuracy results 

    F1 = sum(F1_scores)/k #calculate the average F1 score
    #accuracy = sum(accuracy_scores)/k #calculate the average accuracy score 

    return F1 #, accuracy


# In[9]:


"""this function inputs a number and a dictionary of data sets. Function outputs a training data frame and a testing data frame whose dictionary key corresponds with j 
"""
def kFoldHelper(j, my_dict):
    l = []
    for i in range(len(my_dict)): 
        if i != j: #get all validation sets except the one for testing 
            l += [my_dict[i]] #compile training data 

    train_data = pd.concat(l, ignore_index=True) #concat training data 
    test_data = my_dict[j].reset_index(drop = True)

    return train_data, test_data


# In[90]:


"""function takes as input test_data (data frames). Function implements Naive Bayes classification to predict the response variable. Outputs accuracy, precision, recall, F-1 measure, and a confusion matrix
"""
def performance(test_data, train_data, response):
    #get predicted data 
    predicted_data = classifier(test_data, train_data, response)

    #initialize confusion matrix values
    t_positive = 0
    f_positive = 0
    t_negative = 0
    f_negative = 0

    for i in range(len(test_data)): #for each row in test data

        test_label = test_data.loc[i][-1] #get actual values of the response variable
        
        predicted_label = predicted_data.loc[i][-1] #get predicted values of the response variable 

        if test_label == 1 and predicted_label == 1:
            t_positive += 1
       
        elif test_label == 0 and predicted_label == 1:
            f_positive += 1

        elif test_label == 0 and predicted_label == 0:
            t_negative += 1

        else:
            f_negative += 1
    
    accuracy = (t_positive + t_negative)/(t_positive + t_negative + f_positive + f_negative)
    precision = t_positive/(t_positive + f_positive +0.00001)
    recall = t_positive/(t_positive + f_negative +0.00001) 
    F_1 = 2*precision*recall/(precision + recall +0.00001)

    confusionMatrix = {"C1": [t_positive, f_positive], "C2": [f_negative, t_negative]}
    confusionMatrix = pd.DataFrame(confusionMatrix, index = ["C1", "C2"])

    #print('false pos: ', f_positive)
    #print('false neg: ', f_negative)
    #print('true neg: ', t_negative)
    #print('true pos: ', t_positive)
    #print('accuracy: ', accuracy)
    #print('precision: ', precision)
    #print('F_1: ', F_1)


    return F_1, accuracy, confusionMatrix #, accuracy , precision, recall, 


# In[ ]:




