from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility 为了重现结果

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

from tensorflow.examples.tutorials.mnist import input_data
import PIL.Image as myimage

my_num_data = [];
nb_classes = 10   # 类别

index = [0,2,3,5,9]
for i in index:
    img = myimage.open('/home/yjh/tensorflow/testPIC/'+str(i)+'.png').resize((28,28)).convert('L')
    my_num_data.append(np.array(img))

model = keras.models.load_model('mymodel')
my_num_y = np_utils.to_categorical(index,nb_classes)
my_num_x = np.array(my_num_data).reshape(len(index),784)
my_num_x = my_num_x.astype('float32')
# 取反
my_num_x = 255-my_num_x

# 对每个图片生成预测
print("\ntest:\n")
print(model.predict(my_num_x, batch_size=None, verbose=0, steps=None))

# 生成测试评估
score = model.evaluate(my_num_x, my_num_y, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
