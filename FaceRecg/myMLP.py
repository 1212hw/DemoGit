import numpy



class MyMLP(object):

    # def __init__(self, input, n_in, n_h, n_out):
    #     """ Initialize the parameters of the MLP
    #
    #     :type input: theano.tensor.TensorType
    #     :param input: symbolic variable that describes the input of the
    #                   architecture (one minibatch)
    #
    #     :type n_in: int
    #     :param n_in: number of input units, the dimension of the space in
    #                  which the datapoints lie
    #     :type n_h: int
    #     :param n_h: number of hidden units,
    #
    #     :type n_out: int
    #     :param n_out: number of output units, the dimension of the space in
    #                   which the labels lie
    #
    #     """
    #
    #
    #     # self.W_o = numpy.ones((n_out, n_h), dtype='float64') / (n_out*n_h)
    #     # self.W_h = numpy.ones((n_h, n_in), dtype='float64') / (n_h*n_in)
    #     # self.W_o = numpy.random.rand(n_out, n_h)
    #     # self.W_h = numpy.random.rand(n_h, n_in)
    #     self.W_o = numpy.random.uniform(low=-numpy.sqrt(6. / (n_h + n_out)),
    #                 high=numpy.sqrt(6. / (n_h + n_out)),
    #                 size = (n_out, n_h))
    #     self.W_h = numpy.random.uniform(low=-numpy.sqrt(6. / (n_in + n_h)),
    #                 high=numpy.sqrt(6. / (n_in + n_h)),
    #                 size = (n_h, n_in))
    #     # parameters of the model
    #
    #     # keep track of model input
    #     self.input = input   #[n_in,n_h]
    #     self.input_e = input[numpy.newaxis,:,:]   #扩展为[1,n_in,n_h]
    #
    #     self.z = self.mySoftmax(self.W_h @ self.input)  #[n_h,n_d]
    #     self.z_e = self.z[numpy.newaxis,:,:]    #扩展为[1,n_h,n_d]
    #
    #     self.p_y_given_x = self.mySoftmax(self.W_o @ self.z)  #[n_out,n_d]
    #
    #     # symbolic description of how to compute prediction as class whose
    #     # probability is maximal
    #     self.y_pred = numpy.argmax(self.p_y_given_x, axis=0)
    #     # end-snippet-1
    #
    #
    # def mySoftmax(self, x):
    #     '''
    #     实现理论的softmax函数
    #     x_row:label cnt; x_col:sample cnt
    #     :param x:
    #     :return:
    #     '''
    #     x_array = numpy.array(x)
    #     e_x = numpy.exp(x_array-x_array.max(axis=0,keepdims=True))
    #     x_out = e_x / numpy.sum(e_x,axis=0, keepdims=True)
    #     return x_out
    #
    #
    # def myMLPGradCalcAndParmUpdate(self, input, y, lr, n_in, n_h, n_out, n_d):
    #
    #     # keep track of model input
    #     self.input_e = input[numpy.newaxis,:,:]   #扩展为[1,n_in,n_d]
    #     self.input = input   #[n_in,n_d]
    #
    #     self.z = self.mySoftmax(self.W_h @ self.input)  #[n_h,n_d]
    #     self.z_e = self.z[numpy.newaxis,:,:]    #扩展为[1,n_h,n_d]
    #     self.p_y_given_x = self.mySoftmax(self.W_o @ self.z)  # [n_out,n_d]
    #     # self.y_pred = numpy.argmax(self.p_y_given_x, axis=0)
    #
    #     W_o_GradBuff1 = numpy.zeros((n_out,n_d),dtype='int')
    #     W_o_GradBuff1[y[numpy.arange(n_d)],numpy.arange(n_d)] = 1   #W_o_GradBuff1的第d列的第y[d]行为1
    #     W_o_GradBuff = W_o_GradBuff1 - self.p_y_given_x
    #     W_o_GradBuff_e = W_o_GradBuff[:,numpy.newaxis,:]
    #     W_o_GradVald1 = numpy.sum(W_o_GradBuff[0,:] * self.z[0,:])
    #     W_o_GradVald2 = numpy.sum(W_o_GradBuff[1, :] * self.z[1, :])
    #     W_o_Grad = numpy.sum(self.z_e * W_o_GradBuff_e,2) #[1,n_h,n_d] * [n_out,1,n_d]
    #
    #
    #     W_h_GradCnt = numpy.zeros((n_h, n_d), dtype='float64')
    #     W_h_GradCnt1 = numpy.zeros((n_h, n_d), dtype='float64')
    #     W_h_GradSumBuff1 = numpy.zeros((1, n_d), dtype='float64')
    #     for l_cnt in numpy.arange(n_h):
    #         W_h_GradBuff1 = numpy.ones((n_h, 1), dtype='int')
    #         W_h_GradBuff1[l_cnt] = 0  # W_h_GradBuff1的第l_cnt行为0
    #         W_h_GradBuff2 = (W_h_GradBuff1 - self.z)    #[n_h,n_d]
    #
    #         for d_cnt in numpy.arange(n_d):
    #
    #             #按照输入误差求解
    #             # W_h_GradBuff3 = self.W_o - self.W_o[y[d_cnt], :]  #[n_out,n_h]
    #             # W_h_GradSumBuff1 = numpy.sum(numpy.exp(W_h_GradBuff3 @ self.z) * (self.W_o @ (self.z * W_h_GradBuff2)),
    #             #                              0)  # W_h_GradSumBuff1：[1,n_d]
    #             # W_h_GradSumBuff2 = numpy.sum(numpy.exp(W_h_GradBuff3 @ self.z[:, d_cnt]) * (self.W_o @ (self.z[:, d_cnt] * W_h_GradBuff2[:, d_cnt])),
    #             #                              0)  # W_h_GradSumBuff1：[1]
    #             #
    #             # W_h_GradCntBuff2 = numpy.sum(self.W_o[y[d_cnt],:] * (self.z[:, d_cnt] * W_h_GradBuff2[:, d_cnt]))
    #             # W_h_GradCntBuff1 = numpy.sum(self.W_o[y[d_cnt],:] * (self.z[:, d_cnt] * W_h_GradBuff2[:, d_cnt])) \
    #             #                   - self.p_y_given_x[y[d_cnt],d_cnt] * W_h_GradSumBuff1[d_cnt]  #W_h_GradCntBuff1：[1]
    #             # W_h_GradCntBuff = numpy.sum(self.W_o[y[d_cnt],:] * (self.z[:, d_cnt] * W_h_GradBuff2[:, d_cnt])) \
    #             #                   - self.p_y_given_x[y[d_cnt],d_cnt] * W_h_GradSumBuff2  #W_h_GradCntBuff：[1]
    #             # W_h_GradCnt[l_cnt,d_cnt] = W_h_GradCntBuff
    #
    #             #按照最大误差求解
    #             max_out = numpy.argmax(self.p_y_given_x, axis=0)
    #             W_h_GradBuff3 = self.W_o - self.W_o[max_out[d_cnt], :]  #[n_out,n_h]
    #             W_h_GradSumBuff1 = numpy.sum(numpy.exp(W_h_GradBuff3 @ self.z) * (self.W_o @ (self.z * W_h_GradBuff2)),
    #                                          0)  # W_h_GradSumBuff1：[1,n_d]
    #             W_h_GradSumBuff2 = numpy.sum(numpy.exp(W_h_GradBuff3 @ self.z[:, d_cnt]) * (self.W_o @ (self.z[:, d_cnt] * W_h_GradBuff2[:, d_cnt])),
    #                                          0)  # W_h_GradSumBuff1：[1]
    #
    #             W_h_GradCntBuff2 = numpy.sum(self.W_o[y[d_cnt],:] * (self.z[:, d_cnt] * W_h_GradBuff2[:, d_cnt]))
    #             W_h_GradCntBuff1 = numpy.sum(self.W_o[y[d_cnt],:] * (self.z[:, d_cnt] * W_h_GradBuff2[:, d_cnt])) \
    #                               - self.p_y_given_x[y[d_cnt],d_cnt] * W_h_GradSumBuff1[d_cnt]  #W_h_GradCntBuff1：[1]
    #             W_h_GradCntBuff = numpy.sum(self.W_o[y[d_cnt],:] * (self.z[:, d_cnt] * W_h_GradBuff2[:, d_cnt])) \
    #                               - self.p_y_given_x[max_out[d_cnt],d_cnt] * W_h_GradSumBuff2  #W_h_GradCntBuff：[1]
    #             W_h_GradCnt[l_cnt,d_cnt] = W_h_GradCntBuff
    #
    #         # W_h_GradCntBuff1 = self.W_o[y[numpy.arange(n_d)],:] @ (self.z[:, numpy.arange(n_d)] * \
    #         #     (W_h_GradBuff1 - self.z[:, numpy.arange(n_d)]))
    #         # W_h_GradCnt1[l_cnt,numpy.arange(n_d)] = W_h_GradCntBuff1 - \
    #         #     self.p_y_given_x[y[numpy.arange(n_d)],numpy.arange(n_d)] * W_h_GradSumBuff1[numpy.arange(n_d)]  #W_h_GradCntBuff：[1,n_d]
    #
    #     W_h_GradCnt_e = W_h_GradCnt[:,numpy.newaxis,:]  #[n_h,1,n_d]
    #     W_h_Grad = numpy.sum(self.input_e * W_h_GradCnt_e,2) #[1,n_in,n_h] * [n_h,1,n_d]
    #
    #     self.W_o = self.W_o + lr * W_o_Grad / input.shape[1]        #梯度方向更新
    #     self.W_h = self.W_h + lr * W_h_Grad / input.shape[1]        #梯度方向更新
    #
    # def myMLPPredict(self, input):
    #
    #     self.input_e = input[numpy.newaxis,:,:]   #扩展为[1,n_in,n_h]
    #     self.input = input   #扩展为[1,n_in,n_h]
    #
    #     self.z = self.mySoftmax(self.W_h @ self.input)  #[n_h,n_d]
    #     self.z_e = self.z[numpy.newaxis,:,:]    #扩展为[1,n_h,n_d]
    #     self.p_y_given_x = self.mySoftmax(self.W_o @ self.z)  # [n_out,n_d]
    #     self.y_pred = numpy.argmax(self.p_y_given_x, axis=0)

    '''
    加入偏置
    '''
    def __init__(self, input, n_in, n_h, n_out):
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


        # self.W_o = numpy.ones((n_out, n_h), dtype='float64') / (n_out*n_h)
        # self.W_h = numpy.ones((n_h, n_in), dtype='float64') / (n_h*n_in)
        # self.W_o = numpy.random.rand(n_out, n_h)
        # self.W_h = numpy.random.rand(n_h, n_in)
        self.W_o = numpy.random.randn(n_out, n_h+1)
        self.W_h = numpy.random.randn(n_h, n_in+1)
        # self.W_o = numpy.random.uniform(low=-numpy.sqrt(6. / (n_h + n_out)),
        #             high=numpy.sqrt(6. / (n_h + n_out)),
        #             size = (n_out, n_h+1))
        # self.W_h = numpy.random.uniform(low=-numpy.sqrt(6. / (n_in + n_h)),
        #             high=numpy.sqrt(6. / (n_in + n_h)),
        #             size = (n_h, n_in+1))
        # parameters of the model

        # keep track of model input
        inset_bias = numpy.ones((1,input.shape[1]),dtype='float64')
        self.input = numpy.insert(input,input.shape[0],values=inset_bias,axis=0)   #[n_in+1,n_d]
        self.input_e = self.input[numpy.newaxis,:,:]   #扩展为[1,n_in+1,n_h]

        self.z = numpy.insert(self.mySoftmax(self.W_h @ self.input), n_h, values=inset_bias, axis=0)  #[n_h+1,n_d]
        self.z_e = self.z[numpy.newaxis,:,:]    #扩展为[1,n_h+1,n_d]

        self.p_y_given_x = self.mySoftmax(self.W_o @ self.z)  #[n_out,n_d]

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = numpy.argmax(self.p_y_given_x, axis=0)
        # end-snippet-1


    def mySoftmax(self, x):
        '''
        实现理论的softmax函数
        x_row:label cnt; x_col:sample cnt
        :param x:
        :return:
        '''
        x_array = numpy.array(x)
        e_x = numpy.exp(x_array-x_array.max(axis=0,keepdims=True))
        x_out = e_x / numpy.sum(e_x,axis=0, keepdims=True)
        return x_out


    def myMLPGradCalcAndParmUpdate(self, input, y, lr, n_in, n_h, n_out, n_d):

        # keep track of model input
        inset_bias = numpy.ones((1,input.shape[1]),dtype='float64')
        self.input = numpy.insert(input,input.shape[0],values=inset_bias,axis=0)   #[n_in+1,n_d]
        self.input_e = self.input[numpy.newaxis,:,:]   #扩展为[1,n_in+1,n_h]

        self.z = numpy.insert(self.mySoftmax(self.W_h @ self.input), n_h, values=inset_bias, axis=0)  #[n_h+1,n_d]
        self.z_e = self.z[numpy.newaxis,:,:]    #扩展为[1,n_h+1,n_d]
        self.p_y_given_x = self.mySoftmax(self.W_o @ self.z)  #[n_out,n_d]
        # self.y_pred = numpy.argmax(self.p_y_given_x, axis=0)

        W_o_GradBuff1 = numpy.zeros((n_out,n_d),dtype='int')
        W_o_GradBuff1[y[numpy.arange(n_d)],numpy.arange(n_d)] = 1   #W_o_GradBuff1的第d列的第y[d]行为1
        W_o_GradBuff = W_o_GradBuff1 - self.p_y_given_x
        W_o_GradBuff_e = W_o_GradBuff[:,numpy.newaxis,:]
        W_o_Grad = numpy.sum(self.z_e * W_o_GradBuff_e,2) #[1,n_h+1,n_d] * [n_out,1,n_d]


        W_h_GradCnt = numpy.zeros((n_h, n_d), dtype='float64')
        W_h_GradCnt1 = numpy.zeros((n_h, n_d), dtype='float64')
        W_h_GradSumBuff1 = numpy.zeros((1, n_d), dtype='float64')
        for l_cnt in numpy.arange(n_h):
            W_h_GradBuff1 = numpy.ones((n_h+1, 1), dtype='int')
            W_h_GradBuff1[l_cnt] = 0  # W_h_GradBuff1的第l_cnt行为0
            W_h_GradBuff2 = (W_h_GradBuff1 - self.z)    #[n_h+1,n_d]

            for d_cnt in numpy.arange(n_d):

                #按照输入误差求解
                # W_h_GradBuff3 = self.W_o - self.W_o[y[d_cnt], :]  #[n_out,n_h]
                # W_h_GradSumBuff1 = numpy.sum(numpy.exp(W_h_GradBuff3 @ self.z) * (self.W_o @ (self.z * W_h_GradBuff2)),
                #                              0)  # W_h_GradSumBuff1：[1,n_d]
                # W_h_GradSumBuff2 = numpy.sum(numpy.exp(W_h_GradBuff3 @ self.z[:, d_cnt]) * (self.W_o @ (self.z[:, d_cnt] * W_h_GradBuff2[:, d_cnt])),
                #                              0)  # W_h_GradSumBuff1：[1]
                #
                # W_h_GradCntBuff2 = numpy.sum(self.W_o[y[d_cnt],:] * (self.z[:, d_cnt] * W_h_GradBuff2[:, d_cnt]))
                # W_h_GradCntBuff1 = numpy.sum(self.W_o[y[d_cnt],:] * (self.z[:, d_cnt] * W_h_GradBuff2[:, d_cnt])) \
                #                   - self.p_y_given_x[y[d_cnt],d_cnt] * W_h_GradSumBuff1[d_cnt]  #W_h_GradCntBuff1：[1]
                # W_h_GradCntBuff = numpy.sum(self.W_o[y[d_cnt],:] * (self.z[:, d_cnt] * W_h_GradBuff2[:, d_cnt])) \
                #                   - self.p_y_given_x[y[d_cnt],d_cnt] * W_h_GradSumBuff2  #W_h_GradCntBuff：[1]
                # W_h_GradCnt[l_cnt,d_cnt] = W_h_GradCntBuff

                #按照最大误差求解
                max_out = numpy.argmax(self.p_y_given_x, axis=0)
                W_h_GradBuff3 = self.W_o - self.W_o[max_out[d_cnt], :]  #[n_out,n_h+1]
                W_h_GradSumBuff1 = numpy.sum(numpy.exp(W_h_GradBuff3 @ self.z) * (self.W_o @ (self.z * W_h_GradBuff2)),
                                             0)  # W_h_GradSumBuff1：[1,n_d]
                W_h_GradSumBuff2 = numpy.sum(numpy.exp(W_h_GradBuff3 @ self.z[:, d_cnt]) * (self.W_o @ (self.z[:, d_cnt] * W_h_GradBuff2[:, d_cnt])),
                                             0)  # W_h_GradSumBuff1：[1]

                W_h_GradCntBuff2 = numpy.sum(self.W_o[y[d_cnt],:] * (self.z[:, d_cnt] * W_h_GradBuff2[:, d_cnt]))
                W_h_GradCntBuff1 = numpy.sum(self.W_o[y[d_cnt],:] * (self.z[:, d_cnt] * W_h_GradBuff2[:, d_cnt])) \
                                  - self.p_y_given_x[y[d_cnt],d_cnt] * W_h_GradSumBuff1[d_cnt]  #W_h_GradCntBuff1：[1]
                W_h_GradCntBuff = numpy.sum(self.W_o[y[d_cnt],:] * (self.z[:, d_cnt] * W_h_GradBuff2[:, d_cnt])) \
                                  - self.p_y_given_x[max_out[d_cnt],d_cnt] * W_h_GradSumBuff2  #W_h_GradCntBuff：[1]
                W_h_GradCnt[l_cnt,d_cnt] = W_h_GradCntBuff

            # W_h_GradCntBuff1 = self.W_o[y[numpy.arange(n_d)],:] @ (self.z[:, numpy.arange(n_d)] * \
            #     (W_h_GradBuff1 - self.z[:, numpy.arange(n_d)]))
            # W_h_GradCnt1[l_cnt,numpy.arange(n_d)] = W_h_GradCntBuff1 - \
            #     self.p_y_given_x[y[numpy.arange(n_d)],numpy.arange(n_d)] * W_h_GradSumBuff1[numpy.arange(n_d)]  #W_h_GradCntBuff：[1,n_d]

        W_h_GradCnt_e = W_h_GradCnt[:,numpy.newaxis,:]  #[n_h+1,1,n_d]
        W_h_Grad = numpy.sum(self.input_e * W_h_GradCnt_e,2) #[1,n_in+1,n_h] * [n_h+1,1,n_d]

        self.W_o = self.W_o + lr * W_o_Grad / self.input.shape[1]        #梯度方向更新
        self.W_h = self.W_h + lr * W_h_Grad / self.input.shape[1]        #梯度方向更新

    def myMLPPredict(self, input):

        inset_bias = numpy.ones((1,input.shape[1]),dtype='float64')
        self.input = numpy.insert(input,input.shape[0],values=inset_bias,axis=0)   #[n_in+1,n_d]
        self.input_e = self.input[numpy.newaxis,:,:]   #扩展为[1,n_in+1,n_h]

        self.z = numpy.insert(self.mySoftmax(self.W_h @ self.input), self.W_h.shape[0], values=inset_bias, axis=0)  #[n_h+1,n_d]
        self.z_e = self.z[numpy.newaxis,:,:]    #扩展为[1,n_h+1,n_d]

        self.p_y_given_x = self.mySoftmax(self.W_o @ self.z)  #[n_out,n_d]
        self.y_pred = numpy.argmax(self.p_y_given_x, axis=0)

    def myMLPPredictErr(self, y):

        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        return numpy.mean(self.y_pred != y)