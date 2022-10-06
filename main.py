import pandas as pd
import numpy as np
import math
import tqdm
########## https://www.kaggle.com/code/prashant111/logistic-regression-classifier-tutorial/data
########## logistic regression problem

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y


data = pd.read_csv('weatherAUS.csv')
data = data.fillna(0)

data = data.to_numpy()

## preprocessing
data_for_train = data[:-500,1:-1]
answer_for_train = data[:-500,-2:-1]

data_for_test = data[-500:,1:-1]
answer_for_test = data[-500:-2:-1]


##categories ## !!! I didn't use these
city = sorted(list(set(i[0] for i in data_for_train)))
WindGustDir = sorted(list(set(str(i[6]) for i in data_for_train if str(i[6]) != 'nan')))
WindDir9am = sorted(list(set(str(i[8]) for i in data_for_train if str(i[8]) != 'nan')))
WindDir3pm = sorted(list(set(str(i[9]) for i in data_for_train if str(i[9]) != 'nan')))
RainToday = sorted(list(set(str(i[-1]) for i in data_for_train if str(i[-1]) != 'nan')))

#predict RainTmr - 'Yes' or 'No'// binary classification
RainTmr = sorted(list(set(str(i[-1]) for i in data_for_train if str(i[-1]) != 'nan')))
Y = np.array([1 if i == 'Yes' else 0 for i in answer_for_train[:,-1]])



idx_not_str = []
for i in range(len(data_for_train[0])):
    if type(data_for_train[0][i]) == str or math.isnan(data_for_train[0][i]):
        continue
    else:
        idx_not_str +=[i]

data_numeric = []
for i in idx_not_str:
    data_numeric += [data_for_train[:,i]]  # row = same feature
data_numeric = np.array(data_numeric,dtype=np.float64)


# normalize
for i in range(len(data_numeric)):
    data_numeric[i] = (data_numeric[i] - np.min(data_numeric[i]))  / (np.max(data_numeric[i]) - np.min(data_numeric[i]))



def Gradient(X,W,bias,Y):
    hypothesis = sigmoid(np.dot(X.T, W) + bias)

    #Binary entropy
    H = -np.sum(Y*np.log(hypothesis) + (1-Y)*np.log(1-hypothesis))/len(Y)


    #back propagation
    round_H_round_W = np.zeros((X.shape[0]))
    round_Y_round_W = np.zeros((X.shape))
    round_H_round_bias = np.zeros((X.shape[0]))
    for i in range(X.shape[0]): # 16
        round_Y_round_W[i] = X[i]  # 144960
        round_H_round_W[i] = np.sum((hypothesis - Y) * round_Y_round_W[i]) / len(Y)
    round_H_round_bias =  np.sum((hypothesis - Y) * 1 )/len(Y)

    alpha = 0.1

    W = W - alpha * round_H_round_W
    bias = bias - alpha * round_H_round_bias

    predict = np.round(hypothesis)

    return W, bias, predict, H



np.random.seed(1)
# input matrix
X = data_numeric  #(16,144960)
# parameter matrix
W = np.random.rand(16)
bias = np.random.rand(X.shape[1])
for i in tqdm.tqdm(range(1000)):
    W, bias, predict, H = Gradient(X,W,bias,Y)

    acc = np.sum([1 if predict[i] == Y[i] else 0 for i in range(len(Y))])/len(Y)
    print("acc = {}".format(acc))
    print("H = {}".format(H))

























