import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

np.random.seed(42)

#preparing the data
column_names = ["c1", "c2"]

df1 = pd.read_csv("Group20/Classification/LS_Group20/Class1.txt", sep='\s+', header=None, names=column_names)
df2 = pd.read_csv("Group20/Classification/LS_Group20/Class2.txt", sep='\s+', header=None, names=column_names)
df3 = pd.read_csv("Group20/Classification/LS_Group20/Class3.txt", sep='\s+', header=None, names=column_names) 

df1.insert(0, 'cl0', 1)
df2.insert(0, 'cl0', 1)
df3.insert(0, 'cl0', 1)

train_df1, test_df1 = train_test_split(df1, test_size=0.3)
train_df2, test_df2 = train_test_split(df2, test_size=0.3)
train_df3, test_df3 = train_test_split(df3, test_size=0.3)

train_df1.reset_index(drop=True, inplace=True)
train_df2.reset_index(drop=True, inplace=True)
train_df3.reset_index(drop=True, inplace=True)

test_df1.reset_index(drop=True, inplace=True)
test_df2.reset_index(drop=True, inplace=True)
test_df3.reset_index(drop=True, inplace=True)

epochs = 50

def sigmoidal_func(x):
    return 1/(1 + np.exp(-1*x))

def derivative_sigmoid(x):
    return x*(1-x)

def error(s, label):
    return ((label - s)**2)/2

def epoch_error_plot(y, text):
    x_axis = np.arange(1,len(y)+1)
    plt.plot(x_axis,y)
    plt.xscale('log')
    plt.yscale('log')
    plt.title(text)
    plt.show()

def func_value(train_df1, weights):
    a_n = np.dot(train_df1,weights)
    s = sigmoidal_func(a_n)
    return s

def model_train(train_df1, train_df2):
    weights = np.random.rand(3)
    error_list = []
    for i in range(epochs):
        error_sum = 0
        learning_rate = 1/(i+1)

        for j in range(len(train_df1)):
            s = func_value(train_df1.loc[j], weights)
            error_sum += error(s,1)
            change_weight = np.dot(train_df1.loc[j], ((1 - s)*derivative_sigmoid(s)*learning_rate))
            weights = weights + change_weight

            s = func_value(train_df2.loc[j], weights)
            error_sum += error(s,0)
            change_weight = np.dot(train_df2.loc[j], ((0 - s)*derivative_sigmoid(s)*learning_rate))
            weights = weights + change_weight

        error_list.append(error_sum/(2*len(train_df1)))
    
    return [weights, error_list]

weight_model1, error_model1 = model_train(train_df1, train_df2)
weight_model2, error_model2 = model_train(train_df2, train_df3)
weight_model3, error_model3 = model_train(train_df3, train_df1)

# epoch_error_plot(error_model1, "Epoch vs error for class 1 vs class 2")
# epoch_error_plot(error_model2, "Epoch vs error for class 2 vs class 3")
# epoch_error_plot(error_model3, "Epoch vs error for class 3 vs class 1")

def label_assign(label1, label2, label3):
    labels = [label1, label2, label3]
    max_count = 0
    most_common_number = None

    for number in labels:
        count = labels.count(number)
        if count > max_count:
            max_count = count
            most_common_number = number

    return most_common_number


def test_data(test_df, weight_model1, weight_model2, weight_model3, predicted_label):
    for i in range(len(test_df)):
        s1 = func_value(test_df.loc[i], weight_model1)
        s2 = func_value(test_df.loc[i], weight_model2)
        s3 = func_value(test_df.loc[i], weight_model3)

        label1 = 1 if s1>0.5 else 2
        label2 = 2 if s2>0.5 else 3
        label3 = 3 if s3>0.5 else 1

        predicted_label.append(label_assign(label1, label2, label3))

predicted_label = []
test_data(test_df1, weight_model1, weight_model2, weight_model3, predicted_label)
test_data(test_df2, weight_model1, weight_model2, weight_model3, predicted_label)
test_data(test_df3, weight_model1, weight_model2, weight_model3, predicted_label)

# plt.scatter(test_df1["c1"], test_df1["c2"], color = "blue", label = "class 1")
# plt.scatter(test_df2["c1"], test_df2["c2"], color = "red", label = "class 2")
# plt.scatter(test_df3["c1"], test_df3["c2"], color = "orange", label = "class 3")
# plt.legend()
# plt.show()

given_label = [1]*len(test_df1) + [2]*len(test_df2) + [3]*len(test_df3)

acc_score = accuracy_score(given_label,predicted_label)
conf_matrix = confusion_matrix(given_label,predicted_label)

print(acc_score)
print(conf_matrix)