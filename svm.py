#!/bin/python
import sklearn
from sklearn import svm
from sklearn import semi_supervised
from sklearn.semi_supervised import LabelPropagation
import numpy
from numpy import genfromtxt
from numpy import concatenate

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

def getPrecision(out, test):
    correct = 0.0
    total= 1000.0

    for x in range (0,1000):
        if out[x] == test[x]:
            correct+=1
    return correct/total

# Test 501*5903

def getOuputActiveLearn(train_data, test_data, labels):
    label_prop_model = sklearn.semi_supervised.LabelPropagation(kernel='rbf', gamma=0.00001, n_neighbors=7, alpha=1, max_iter=50, tol=0.0001)
    label_prop_model.fit(train_data, labels)
    out = label_prop_model.predict(test_data)
    return out.astype(int)

def getOuputActiveLearn2(train_data, test_data, labels):
    label_prop_model = sklearn.semi_supervised.LabelPropagation(kernel='knn', gamma=20, n_neighbors=6, alpha=1, max_iter=50, tol=0.0001)
    label_prop_model.fit(train_data, labels)
    out = label_prop_model.predict(test_data)
    return out.astype(int)

def getMinus1():
    out2 = numpy.array([])
    for i in range(0,1000):
        out2=numpy.hstack((out2, numpy.array([-1])))
    return out2

def part3():
    # Training 501*5903
    C = 30
    theta = pow(10,-5)

    label_file = 'lable_1000.csv'
    label_test = genfromtxt(label_file, delimiter=',')

    # Part I flow
    train_file = 'train_group_no_header.csv'
    test_file = 'test_group.csv'
    col_x = 5903
    col_y = 5907

    train_data = genfromtxt(train_file, delimiter=',', usecols=range(0, col_x))
    train_output = genfromtxt(train_file, delimiter=',', usecols=range(col_x, col_y))
    test_data = genfromtxt(test_file, delimiter=',')

    #centralize train and test
    train_data = train_data - numpy.mean(train_data, axis=0)
    test_data = test_data - numpy.mean(test_data, axis=0)


    out = getOutput(train_data, test_data, train_output[:, 0], C, theta)

    precision = getPrecision(out, label_test)
    print "train data shape: "+str(train_data.shape)
    print "first precision: "+str(precision)

    # use 1000 new data as test => out2
    part2_1000_file_1 = 'provideData.csv'
    part2_1000_file_2 = 'provideData2.csv'
    part2_1 = genfromtxt(part2_1000_file_1, delimiter=',')
    part2_2 = genfromtxt(part2_1000_file_2, delimiter=',')
    part2 = concatenate((part2_1, part2_2), axis=1)
    part2 = part2 - numpy.mean(part2, axis=0)

    # out2 = getOutput(train_data, part2, train_output[:, 0], C, theta)
    out2 = getMinus1()

    # use 1500 new_data to train and test again.
    new_train_data = concatenate((train_data, part2), axis=0)
    new_output =  concatenate((train_output[:,0], out2), axis=0)
    new_out = getOuputActiveLearn(new_train_data,test_data, new_output)

    #
    # new_out = getOutput(new_train_data, test_data, new_output, C, theta)
    #
    precision2 = getPrecision(new_out, label_test)
    print "train data shape: "+str(new_train_data.shape)
    print "second precision: "+str(precision2)



    new_out2 = getOuputActiveLearn2(new_train_data,test_data, new_output)
    precision3 = getPrecision(new_out2, label_test)
    print "train data shape: "+str(new_train_data.shape)
    print "third precision: "+str(precision3)

def test():
    train_file = 'train_group_no_header.csv'
    col_x = 5903
    col_y = 5907
    train_output = genfromtxt(train_file, delimiter=',', usecols=range(col_x, col_y))
    #print train_output[:,0]
    print train_output[:,0].shape
    out2 = getMinus1()
    #print out2
    print out2.shape
    new_output =  concatenate((train_output[:,0], out2), axis=0)
    print new_output.shape
#test()
part3()
#text = process_data(out)
# w_name = "pricision.csv"
# with open(w_name, "w") as text_file:
#     text_file.write(text)


# Part I flow
# Train as 500, Test use Part II data => label, precision

# Train as 1500, Test use Test => label, precision


