#!/bin/python
from sklearn import svm
import numpy
from numpy import genfromtxt


def getOutput(train_data, test_data, train_output_col, c, gama):
    "This prints a passed string into this function"
    clf = svm.SVC(C=c, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape=None, degree=3, gamma=gama, kernel='rbf',
                  max_iter=-1, probability=False, random_state=None, shrinking=True,
                  tol=0.001, verbose=False)
    clf.fit(train_data, train_output_col)
    out = clf.predict(test_data)
    return out.astype(int)


def getOutput_prob(train_data, test_data, train_output_col, c):
    "This prints a passed string into this function"
    clf = svm.SVC(C=c, cache_size=200, class_weight=None, coef0=0.0,
                  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
                  max_iter=-1, probability=True, random_state=None, shrinking=False,
                  tol=0.001, verbose=False)
    clf.fit(train_data, train_output_col)
    out = clf.predict_proba(test_data)
    return out


def process_data(data):
    text = ""
    for out in data:
        if (out == 0):
            text += "1.0,0.0,0.0\n"
        elif (out == 1):
            text += "0.0,1.0,0.0\n"
        else:
            text += "0.0,0.0,1.0\n"
    return text.rstrip()


# Test 501*5903

# Training 501*5903

train_file = 'train_group_no_header.csv'
test_file = 'test_group.csv'
col_x = 5903
col_y = 5907

train_data = genfromtxt(train_file, delimiter=',', usecols=range(0, col_x))
train_output = genfromtxt(train_file, delimiter=',', usecols=range(col_x, col_y))
test_data = genfromtxt(test_file, delimiter=',')

#
# out = getOutput_prob(train_data,test_data,train_output[:,0],50)
#
# print out
# numpy.savetxt("out.csv", out, delimiter=",")

for i in range(20, 50, 5):
    for j in range(1, 10):
        print("doing pricision" + `i` + `j` + ".csv")
        out = getOutput(train_data, test_data, train_output[:, 0], i, pow(10, -j))
        text = process_data(out)
        w_name = "pricision" + `i` + `j` + ".csv"
        with open(w_name, "w") as text_file:
            text_file.write(text)
            # print()

print('hello')




# for i in range (1,100):
#   out = getOutput(train_data,test_data,train_output[:,0],i)
#   # print out
#   text = process_data(out)
#   w_name="pricision"+`i`+".csv"
#   with open(w_name, "w") as text_file:
#     text_file.write(text)

# numpy.savetxt("out.csv", text, delimiter=",")
# print text
