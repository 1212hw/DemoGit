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
import myLogisticRegression
import timeit
import sys
import os
import matplotlib.pyplot as plt


dataset = r'D:\hewei\program\python\FaceRecgData\mnist.pkl.gz'
learning_rate = 0.3
n_epochs = 100
batch_size = 600
output_len = 10

input = numpy.ones((4,2), dtype='int')
input_t = input.T
y = numpy.array([[1],[2],[3],[4]])
p_y_given_x = numpy.array([[1,2,3],[2,3,4],[3,4,5],[5,6,7]])
p_y_given_x_x = numpy.array([p_y_given_x[numpy.arange(y.shape[0]), 1]]).T
p_y_given_x_x_x = (y == 1)
W_T1 = input_t @ (p_y_given_x_x_x - p_y_given_x_x)
W_T2 = numpy.dot(input_t , (p_y_given_x_x_x - p_y_given_x_x))




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

cc = train_set_x[0]
cc = cc.reshape(28,28)
# python要用show展现出来图
fig = plt.figure()
ax = fig.add_subplot(1,2,1)  #第一个子图
ax.imshow(cc)   #默认配置
plt.show()   #显示图像

tarin_x = train_set_x[0:batch_size]
tarin_y = train_set_y[0:batch_size]
classifier = myLogisticRegression.MyLogisticRegression(input=tarin_x, n_in=n_train_point, n_out=output_len)
# the cost we minimize during training is the negative log likelihood of
# the model in symbolic format
# cost = classifier.myNegative_log_likelihood(tarin_y)
classifier.myNegative_log_likelihoodGrad(input=tarin_x, y=tarin_y,lr = learning_rate)

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

best_validation_loss = numpy.inf
test_score = 0.
start_time = timeit.default_timer()

done_looping = False
epoch = 0

while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            tarin_x = train_set_x[(minibatch_index*batch_size):((minibatch_index+1)*batch_size)]
            tarin_y = train_set_y[(minibatch_index*batch_size):((minibatch_index+1)*batch_size)]
            classifier.myLogisticRegressionCalc(input=tarin_x)
            # the cost we minimize during training is the negative log likelihood of
            # the model in symbolic format
            classifier.myNegative_log_likelihoodGrad(input=tarin_x, y=tarin_y, lr=learning_rate)

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = numpy.arange(0,n_valid_batches,dtype='float64')
                for i in range(n_valid_batches):
                    valid_x = valid_set_x[(i * batch_size):((i + 1) * batch_size)]
                    valid_y = valid_set_y[(i * batch_size):((i + 1) * batch_size)]
                    classifier.myLogisticRegressionCalc(valid_x)
                    temp = classifier.myLogisticErors(valid_y)
                    validation_losses[i] = temp

                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = numpy.arange(0,n_test_batches,dtype='float64')
                    for i in range(n_test_batches):
                        test_x = test_set_x[(i * batch_size):((i + 1) * batch_size)]
                        test_y = test_set_y[(i * batch_size):((i + 1) * batch_size)]
                        classifier.myLogisticRegressionCalc(test_x)
                        test_losses[i] = classifier.myLogisticErors(test_y)

                    test_score = numpy.mean(test_losses)

                    print(
                        (
                            '     epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%'
                        ) %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.
                        )
                    )

                    # save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

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




