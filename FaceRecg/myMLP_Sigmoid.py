import numpy


'''
时间比hulk的长，第一代正确率差不多
'''
class MyMLP_Sigmoid(object):
    def __init__(self, n_in, n_h, n_out):
        """ Initialize the parameters of the MLP

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie
        :type n_h: int
        :param n_h: number of hidden units,

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        #随机正态分布初始化，初始化错误率50%，一代能到10%
        self.W_o = numpy.random.randn(n_out, n_h)
        self.B_o = numpy.random.randn(n_out, 1)

        self.W_h = numpy.random.randn(n_h, n_in)
        self.B_h = numpy.random.randn(n_h, 1)

        # # 随机初始化，初始化错误率90%，并不再下降
        # self.W_o = numpy.random.rand(n_out, n_h)
        # self.B_o = numpy.random.rand(n_out, 1)
        #
        # self.W_h = numpy.random.rand(n_h, n_in)
        # self.B_h = numpy.random.rand(n_h, 1)
        # parameters of the model

    #### Miscellaneous functions
    def sigmoid(self,z):
        """The sigmoid function."""
        return 1.0/(1.0+numpy.exp(-z))

    def myMLPGradCalcAndParmUpdate(self, input, y, lr, n_out, n_d):
        #依据输入，将其序列化，得到[0,1,0,...1]向量

        self.z = self.sigmoid((self.W_h @ input) + self.B_h)  #[n_h,n_d]
        self.z_e = self.sigmoid(self.W_h @ input + self.B_h)  # [n_h,n_d]
        self.z_e = self.z[numpy.newaxis,:,:]    #扩展为[1,n_h,n_d]

        self.p_y_given_x = self.sigmoid(self.W_o @ self.z + self.B_o)  #[n_out,n_d]
        # self.p_y_given_x_e = self.p_y_given_x[numpy.newaxis,:,:]    #扩展为[1,n_out,n_d]

        # 输出层参数
        B_o_GradBuff = (y-self.p_y_given_x) * self.p_y_given_x * (1 - self.p_y_given_x)    #[n_out,n_d]
        B_o_Grad = numpy.sum(B_o_GradBuff,axis=1,keepdims=True)

        B_o_GradBuff = B_o_GradBuff[:,numpy.newaxis,:]   #扩展为[n_out,1,n_d]
        W_o_GradBuff = B_o_GradBuff * self.z_e      #[n_out,n_h,n_d]
        W_o_Grad = numpy.sum(W_o_GradBuff,2)        #[n_out,n_h]

        self.W_o = self.W_o + lr * W_o_Grad / n_d        #梯度方向更新
        self.B_o = self.B_o + lr * B_o_Grad / n_d

        # 隐藏层参数
        W_o_e = self.W_o[:,:,numpy.newaxis]      #[n_out,n_h,1]
        B_h_GradBuff = W_o_GradBuff * (1-self.z_e) * W_o_e     #[n_out,n_h,n_d]
        B_h_Grad = numpy.sum(numpy.sum(B_h_GradBuff,0),axis=1,keepdims=True)

        input_e = input[numpy.newaxis,numpy.newaxis,:,:]    #扩展为[1,1,n_in,n_d]
        B_h_GradBuff = B_h_GradBuff[:,:,numpy.newaxis,:]    #扩展为[n_out,n_h,1,n_d]
        W_h_GradBuff = B_h_GradBuff * input_e
        W_h_Grad = numpy.sum(numpy.sum(W_h_GradBuff,0),2)

        self.W_h = self.W_h + lr * W_h_Grad / n_d        #梯度方向更新
        self.B_h = self.B_h + lr * B_h_Grad / n_d


    def myMLPPredict(self, input):

        self.z = self.sigmoid(self.W_h @ input + self.B_h)  #[n_h,n_d]
        self.p_y_given_x = self.sigmoid(self.W_o @ self.z + self.B_o)  #[n_out,n_d]
        self.y_pred = numpy.argmax(self.p_y_given_x, axis=0)

    def myMLPPredictErr(self, y):

        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        return numpy.mean(self.y_pred != y)