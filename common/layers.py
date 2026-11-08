import numpy as np
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
    x = np.array( [[1.0, -0.5], [-2.0, 3.0]])
    print(x)
    mask = (x <= 0)
    print(mask)


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        # 입력 데이터의 원래 형태 저장
        self.original_x_shape = x.shape
        # 입력 데이터를 2차원 배열로 변환
        x = x.reshape(x.shape[0], -1)
        self.x = x
        # 순전파 계산
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        # 입력 데이터에 대한 기울기
        dx = np.dot(dout, self.W.T)
        # 가중치에 대한 기울기
        self.dW = np.dot(self.x.T, dout)
        # 편향에 대한 기울기
        self.db = np.sum(dout, axis=0)
        # 원래 입력 형태로 복원
        dx = dx.reshape(*self.original_x_shape)
        return dx


def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        y = np.exp(x)
        return y / np.sum(y, axis=1, keepdims=True)

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx * dout
