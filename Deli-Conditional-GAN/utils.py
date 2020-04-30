import os
import numpy as np
import scipy
import scipy.misc
import matplotlib.pyplot as plt


class Mnist(object):

    def __init__(self):

        self.dataname = "Mnist"
        self.dims = 28*28
        self.shape = [28 , 28 , 1]
        self.image_size = 28
        self.data, self.data_y = self.load_mnist()

    def load_mnist(self):

        # data_dir = os.path.join("./data", "mnist") #连接两个或更多的路径名组件
        data_dir = "MNIST_data"
        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd , dtype=np.uint8)  #从文件中读出
        trX = loaded[16:].reshape((60000, 28 , 28 ,  1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28 , 28 , 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)  #将结构数据转换为ndarray类型
        teY = np.asarray(teY)

        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0)
      
        seed = 547

        np.random.seed(seed)  #seed相同，产生的随机数相同
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        #convert label to one-hot

        # # 在这里固定住训练集
        # data_size = 50

        y_vec = np.zeros((len(y), 10), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, int(y[i])] = 1.0

        return X / 255., y_vec

    def getNext_batch(self, iter_num=0, batch_size=64):

        # ro_num = 640 / batch_size - 1  # 要改

        ro_num = len(self.data) / batch_size - 1

        if iter_num % ro_num == 0:  # 为了每次batch输入的数据不同
            length = len(self.data)
            perm = np.arange(length) # 随机
            np.random.shuffle(perm)
            self.data = np.array(self.data)
            self.data = self.data[perm]
            self.data_y = np.array(self.data_y)
            self.data_y = self.data_y[perm]

        return self.data[int(iter_num % ro_num) * batch_size: int(iter_num% ro_num + 1) * batch_size] \
            , self.data_y[int(iter_num % ro_num) * batch_size: int(iter_num%ro_num + 1) * batch_size]


def get_image(image_path , is_grayscale = False):
    return np.array(inverse_transform(imread(image_path, is_grayscale)))


def save_images(images , size , image_path):
    if images.shape[0] != size[0]*size[1]:
        ran = range(size[0]*size[1], images.shape[0])
        images = np.delete(images, ran, axis=0)
    return imsave(inverse_transform(images) , size , image_path)


def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images , size , path):
    return scipy.misc.imsave(path , merge(images , size))

#size [8,8]    ;images.shape(28,28),将一个batch输出64张图片变成8*8排列
def merge(images , size):
    h , w = images.shape[1] , images.shape[2]
    img = np.zeros((h*size[0] , w*size[1] , 3))
    for idx , image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h +h , i*w : i*w+w , :] = image

    return img

def inverse_transform(image):
    return (image + 1.)/2.

def read_image_list(category):
    filenames = []
    print("list file")
    list = os.listdir(category)

    for file in list:
        filenames.append(category + "/" + file)

    print("list file ending!")

    return filenames

##from caffe
def vis_square(visu_path , data , type):
    """Take an array of shape (n, height, width) or (n, height, width , 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))

    padding = (((0, n ** 2 - data.shape[0]) ,
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data , padding, mode='constant' , constant_values=1)  # pad with ones (white)

    # tilethe filters into an im age
    data = data.reshape((n , n) + data.shape[1:]).transpose((0 , 2 , 1 , 3) + tuple(range(4 , data.ndim + 1)))

    data = data.reshape((n * data.shape[1] , n * data.shape[3]) + data.shape[4:])

    plt.imshow(data[:,:,0])
    plt.axis('off')

    if type:
        plt.savefig('./{}/weights.png'.format(visu_path) , format='png')
    else:
        plt.savefig('./{}/activation.png'.format(visu_path) , format='png')


def sample_label():
    num = 64
    label_vector = np.zeros((num , 10), dtype=np.float)
    for i in range(0 , num):
        label_vector[i , int(i/8)] = 1.0
    return label_vector

def sample_10_label():
    num = 64
    label_vector = np.zeros((num , 10), dtype=np.float)
    for i in range(0 , num):
        label_vector[i , int(i/6)%10] = 1.0
    return label_vector