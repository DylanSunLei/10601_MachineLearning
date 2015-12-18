__author__ = 'xinxinpan'

from sklearn import linear_model
from numpy import genfromtxt
import numpy


def getOutput(train_data, train_output_col, test_data):
    "This prints a passed string into this function"
    clf = linear_model.Lasso(alpha=0.05)
    clf.fit(train_data, train_output_col)
    another = clf.predict(test_data)
    return another


# Test 2731*5903

# Training 3172*5903

train_file = 'inputdata.csv'
train_targe_file="outputdata.csv"
test_file = 'test_data.csv'
col_x = 3172
col_y = 2730

train_data = genfromtxt(train_file, delimiter=',')
train_output = genfromtxt(train_targe_file, delimiter=',')
test_data = genfromtxt(test_file, delimiter=',')


totaloutput=numpy.zeros((1000,2731))
#out = getOutput(train_data, train_output[:,0], test_data)
#totaloutput=out

#csvwrite(text,'output.csv')
#numpy.savetxt("out.csv", totaloutput, delimiter=",")
for i in range(0, 2731):
    out = getOutput(train_data, train_output[:, i], test_data)
    totaloutput[:, i]=out


numpy.savetxt("lassoout.csv", totaloutput, delimiter=",")
# print text
