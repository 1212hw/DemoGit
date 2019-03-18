"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""


import numpy


def mySoftmax(x):
    '''
    实现理论的softmax函数
    x_row:sample cnt; x_col:label cnt
    :param x:
    :return:
    '''
    x_array = numpy.array(x)
    e_x = numpy.exp(x_array-x_array.max(axis=1,keepdims=True))
    x_out = e_x / numpy.sum(e_x,axis=1, keepdims=True)
    return x_out


class MyLogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)

        self.W = numpy.zeros((n_in, n_out), dtype='float64')
        self.b = numpy.zeros((1,n_out), dtype='float64')
        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        y_sim = numpy.dot(input,self.W)
        y_sim += self.b
        self.p_y_given_x = mySoftmax(y_sim)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = numpy.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

    def myLogisticRegressionCalc(self, input):
        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        y_sim = numpy.dot(input,self.W)
        y_sim += self.b
        self.p_y_given_x = mySoftmax(y_sim)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = numpy.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

    def myNegative_log_likelihoodGrad(self, input, y, lr):
        '''
        实现理论的softmax梯度
        x_row:sample cnt; x_col:label cnt
        :param x:
        :return:
        '''
        W_T = numpy.zeros((self.W.shape[0],self.W.shape[1]),dtype='float64')
        b_T = numpy.zeros((1,self.W.shape[1]), dtype='float64')
        for i in numpy.arange(self.W.shape[1]):
            W_buff = numpy.array([(y == i) - (self.p_y_given_x[numpy.arange(y.shape[0]), i])]).T
            b_T[0, i] = numpy.mean(W_buff)
            W_buff1 = input.T @ W_buff
            W_T[:,i] = W_buff1.T
        self.W = self.W + W_T*lr/input.shape[0]
        self.b = self.b + b_T*lr

    def myNegative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -numpy.mean(numpy.log(self.p_y_given_x)[numpy.arange(y.shape[0]), y])
        # end-snippet-2

    def myLogisticErors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        return numpy.mean(self.y_pred != y)
        # if y.dtype.isinstance:
        #     # the T.neq operator returns a vector of 0s and 1s, where 1
        #     # represents a mistake in prediction
        #     return numpy.mean(self.y_pred != y)
        # else:
        #     raise NotImplementedError()