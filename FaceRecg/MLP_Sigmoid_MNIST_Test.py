'''
http://deeplearning.net/tutorial/contents.html
Classifying MNIST digits using Logistic Regression
__author__='HW'
__data__='20190318'



'''

import numpy
import cv2
import gzip
import six.moves.cPickle as pickle
import myGeneralMLP_Sigmoid
import timeit
import sys
import os
import matplotlib.pyplot as plt

test1 = numpy.ones((2,5),dtype='int')
test2 = numpy.exp(test1)
test3 = 1 - test1


dataset = r'D:\hewei\program\python\FaceRecgData\mnist.pkl.gz'
learning_rate = 3
n_epochs = 100
batch_size = 10
output_len = 10
hidden_len = 30



with gzip.open(dataset, 'rb') as f:
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    except:
        train_set, valid_set, test_set = pickle.load(f)

test_set_x, test_set_y = train_set
valid_set_x, valid_set_y = valid_set
train_set_x, train_set_y = train_set

# compute number of minibatches for training, validation and testing
n_train_batches,n_train_point = numpy.shape(train_set_x)
n_valid_batches,n_valid_point = numpy.shape(valid_set_x)
n_test_batches,n_test_point = numpy.shape(test_set_x)

n_train_batches = n_train_batches // batch_size
n_valid_batches = n_valid_batches // batch_size
n_test_batches = n_test_batches // batch_size


#数据格式对齐
test_set_x = test_set_x.T
valid_set_x = valid_set_x.T
train_set_x = train_set_x.T

test_set_y = test_set_y.T
valid_set_y = valid_set_y.T
train_set_y = train_set_y.T

# 依据输入，将其序列化，得到[0,1,0,...1]向量
train_set_y_buff = numpy.zeros((output_len, train_set_y.shape[0]), dtype='int32')
train_set_y_buff[train_set_y[numpy.arange(train_set_y.shape[0])], numpy.arange(train_set_y.shape[0])] = 1


# cc = train_set_x[0]
# cc = cc.reshape(28,28)
# # python要用show展现出来图
# fig = plt.figure()
# ax = fig.add_subplot(1,2,1)  #第一个子图
# ax.imshow(cc)   #默认配置
# plt.show()   #显示图像

# classifier = myMLP_Sigmoid.MyMLP_Sigmoid(n_in=n_train_point, n_h=hidden_len, n_out=output_len)
mlpLayers = numpy.array([n_train_point, hidden_len, output_len], dtype='int32')
classifier = myGeneralMLP_Sigmoid.MyGeneralMLP_Sigmoid(mlpLayers, n_d=batch_size)

###############
# TRAIN MODEL #
###############
print('... training the model')
# early-stopping parameters
patience = 5000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
# found
improvement_threshold = 0.995  # a relative improvement of this much is
# considered significant
validation_frequency = min(n_train_batches, patience // 10)
# go through this many
# minibatche before checking the network
# on the validation set; in this case we
# check every epoch

test_score = 0.
start_time = timeit.default_timer()

done_looping = False
epoch = 0

best_validation_loss = 0
while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):
        tarin_x = train_set_x[:, (minibatch_index*batch_size):((minibatch_index+1)*batch_size)]
        tarin_y = train_set_y_buff[:, (minibatch_index*batch_size):((minibatch_index+1)*batch_size)]
        # classifier.myMLPGradCalcAndParmUpdate(input=tarin_x, y=tarin_y, lr=learning_rate,
        #                                       n_out=output_len, n_d=batch_size)
        classifier.myMLPGradCalcAndParmUpdate(input=tarin_x, y=tarin_y, lr=learning_rate,
                                              n_d=batch_size)

    valid_x = valid_set_x
    valid_y = valid_set_y
    classifier.myMLPPredict(valid_x)
    temp = classifier.myMLPPredictErr(valid_y)

    if best_validation_loss < temp:
        best_validation_loss = temp
    print(
        'epoch %i, validation error %f %%' %
        (
            epoch,
            temp * 100.
        )
    )


test_x = test_set_x
test_y = test_set_y
classifier.myMLPPredict(test_x)
test_score = classifier.myMLPPredictErr(test_y)


end_time = timeit.default_timer()
print(
        (
            'Optimization complete with best validation score of %f %%,'
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., test_score * 100.)
    )
print('The code run for %d epochs, with %f epochs/sec' % (
    epoch, 1. * epoch / (end_time - start_time)))
print(('The code for file ' +
       os.path.split(__file__)[1] +
       ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)




