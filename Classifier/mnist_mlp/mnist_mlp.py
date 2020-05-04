'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.

在MNIST数据集上训练一个简单的深度神经网络。
在20轮迭代后获得了 98.40% 的测试准确率
(还有很多参数调优的余地)。
在 K520 GPU上，每轮迭代 2 秒。
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility 为了重现结果

from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
import PIL.Image as Image
import os

# 模型参数
batch_size = 128  # 批大小
nb_classes = 10   # 类别
nb_epoch = 20     # 迭代轮次

# 读取原始数据集  只读取测试集
# the data, shuffled and split between train and test sets
# 经过打乱和切分的训练与测试数据
(_, _), (X_test, y_test) = mnist.load_data()
# (_, _), (X_test, y_test) = fashion_mnist.load_data()

# 读取自己的数据集
simple_x = []
simple_y = []
fake_x = []
fake_y = []

# 读取当前目录下的所有文件
simple_path = 'datasets/mnist_part'
# simple_path = 'datasets/fashion-mnist_part'
for file in os.listdir(simple_path):
    simple_y.append(int(file.split('_')[0]))
    img = Image.open(simple_path+'/'+file).resize((28,28)).convert('L')
    simple_x.append(np.array(img))

fake_path = 'generate-images/mnist_new96'
# fake_path = 'generate-images/fashion_new32'
for file in os.listdir(fake_path):
    fake_y.append(int(file.split('_')[0]))
    img = Image.open(fake_path+'/'+file).resize((28,28)).convert('L')
    fake_x.append(np.array(img))

X_train = np.concatenate([simple_x, fake_x], axis=0)
y_train = simple_y + fake_y

# # 只测试小样本训练集
# X_train = np.array(simple_x)
# y_train = simple_y

# 将 28x28 的图像展平成一个 784 维的向量
X_train = X_train.reshape(-1, 784)
X_train = X_train.astype('float32')
X_test = X_test.reshape(-1, 784)
X_test = X_test.astype('float32')
# 规范化
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# 将分类向量转化为二元分类矩阵(one hot)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# 随机一下
seed = 500
np.random.seed(seed)  # seed相同，产生的随机数相同
np.random.shuffle(X_train)
np.random.seed(seed)
np.random.shuffle(Y_train)


# 模型定义
model = Sequential()  # 序列化模型
model.add(Dense(1024, input_shape=(784,)))  # 全连接层，784维输入，512维输出
model.add(Activation('relu'))              # relu 激活函数
# model.add(Dropout(0.2))                    # Dropout 舍弃部分因曾结点的权重，防止过拟合
model.add(Dense(256))                      # 由一个全连接层，512维输出
model.add(Activation('relu'))              # relu 激活函数
# model.add(Dropout(0.2))                    # Dropout 舍弃部分因曾结点的权重，防止过拟合
model.add(Dense(10))                       # 由一个全连接层，10维输出
model.add(Activation('softmax'))           # softmax 激活函数用于分类
model.summary()                            # 模型概述

# 定义模型的损失函数，优化器，评估矩阵
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# 训练，迭代 nb_epoch 次
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1)

# 评估测试集测试误差，准确率
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])