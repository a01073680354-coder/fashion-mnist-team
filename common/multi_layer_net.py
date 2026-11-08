# coding: utf-8
# GitHub: https://github.com/WegraLee/deep-learning-from-scratch/blob/master/common/multi_layer_net.py
# 수업 코드에서 사용한 계층으로 구성
import numpy as np
from collections import OrderedDict
from common.layers import Affine, Relu, SoftmaxWithLoss


class MultiLayerNet:
    """ReLU를 사용하는 다층 신경망"""

    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # He 초기화
        all_sizes = [input_size] + hidden_size_list + [output_size]
        for i in range(1, len(all_sizes)):
            scale = np.sqrt(2.0 / all_sizes[i - 1])
            self.params['W' + str(i)] = scale * np.random.randn(all_sizes[i - 1], all_sizes[i])
            self.params['b' + str(i)] = np.zeros(all_sizes[i])

        # 계층 생성
        self.layers = OrderedDict()
        for i in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(i)] = Affine(self.params['W' + str(i)], self.params['b' + str(i)])
            self.layers['Relu' + str(i)] = Relu()

        i = self.hidden_layer_num + 1
        self.layers['Affine' + str(i)] = Affine(self.params['W' + str(i)], self.params['b' + str(i)])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        weight_decay = 0
        for i in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(i)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = np.argmax(self.predict(x), axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.mean(y == t)

    def gradient(self, x, t):
        # 오차역전파로 기울기 계산
        self.loss(x, t)
        dout = self.last_layer.backward(1)
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        grads = {}
        for i in range(1, self.hidden_layer_num + 2):
            affine = self.layers['Affine' + str(i)]
            grads['W' + str(i)] = affine.dW + self.weight_decay_lambda * affine.W
            grads['b' + str(i)] = affine.db
        return grads
