import os, time, itertools, imageio, pickle, random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 100

# leaky_relu
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

# 小批量判别器  用于解决模式崩溃的问题   直接计算批量样本的统计特征
def Minibatch_Discriminator(input, num_kernels=100, dim_per_kernel=5, init=False, name='MD'):
    num_inputs=16*4
    theta = tf.get_variable(name+"/theta",[num_inputs, num_kernels, dim_per_kernel], initializer=tf.random_normal_initializer(stddev=0.05))
    log_weight_scale = tf.get_variable(name+"/lws",[num_kernels, dim_per_kernel], initializer=tf.constant_initializer(0.0))
    W = tf.multiply(theta, tf.expand_dims(tf.exp(log_weight_scale)/tf.sqrt(tf.reduce_sum(tf.square(theta),0)),0))
    W = tf.reshape(W,[-1,num_kernels*dim_per_kernel])
    x = input
    x=tf.reshape(x, [batch_size,num_inputs])
    activation = tf.matmul(x, W)
    activation = tf.reshape(activation,[-1,num_kernels,dim_per_kernel])
    abs_dif = tf.multiply(tf.reduce_sum(tf.abs(tf.subtract(tf.expand_dims(activation,3),tf.expand_dims(tf.transpose(activation,[1,2,0]),0))),2),
                                                1-tf.expand_dims(tf.constant(np.eye(batch_size),dtype=np.float32),1))
    f = tf.reduce_sum(tf.exp(-abs_dif),2)/tf.reduce_sum(tf.exp(-abs_dif))
    print((f.get_shape()))
    print((input.get_shape()))
    return tf.concat(axis=1,values=[x, f])

# G(z)  x : ? 1 1 100  y : ? 1 1 10
def generator(x, y_label, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # concat layer  链接某个维度的层   在第3个维度链接  相当于是颜色那一层 
        # 维度 x : ? 1 1 100    y : ? 1 1 10
        # cat1 -> ? 1 1 110
        cat1 = tf.concat([x, y_label], 3)  

        # ? 4 4 64
        fc1 = tf.reshape(tf.layers.dense(cat1, 1152, kernel_initializer=w_init), [-1, 3, 3, 128])

        # 1st hidden layer 
        # decov1: ? 7 7 32
        deconv1 = tf.layers.conv2d_transpose(fc1, 32, [3, 3], strides=(2, 2), padding='valid', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu1 = lrelu(tf.layers.batch_normalization(deconv1, training=isTrain), 0.2)

        # 2nd hidden layer
        # ? 14 14 16
        deconv2 = tf.layers.conv2d_transpose(lrelu1, 16, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu2 = lrelu(tf.layers.batch_normalization(deconv2, training=isTrain), 0.2)

        # output layer
        # 输出 ? 28 28 1
        deconv3 = tf.layers.conv2d_transpose(lrelu2, 1, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        o = tf.nn.tanh(deconv3)

        return o

# D(x)   x : ? 28 28 1  y : ? 28 28 10    reuse 是重复使用的意思
def discriminator(x, y_fill, isTrain=True, reuse=False):   
    with tf.variable_scope('discriminator', reuse=reuse):
        # initializer
        w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        b_init = tf.constant_initializer(0.0)

        # concat layer  在第3个维度链接
        cat1 = tf.concat([x, y_fill], 3)

        # 1st hidden layer
        # 14 14 16
        conv1 = tf.layers.conv2d(cat1, 16, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu1 = lrelu(conv1, 0.2)

        # 2nd hidden layer
        # 7 7 32
        conv2 = tf.layers.conv2d(lrelu1, 32, [5, 5], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3nd hidden layer
        # 4 4 64
        conv3 = tf.layers.conv2d(lrelu2, 64, [3, 3], strides=(2, 2), padding='same', kernel_initializer=w_init, bias_initializer=b_init)
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        h4 = tf.layers.max_pooling2d(lrelu3, [4, 4], [1, 1])

        h5 = Minibatch_Discriminator(h4, 128, name='discriminator_MD')

        h6 = tf.reshape(tf.layers.dense(h5, 1),[-1, 1, 1, 1])
        # output layer
        # 1 1 1
        o = tf.nn.sigmoid(h6)

        return o, h6

# preprocess  预处理   先随机生成10组数据  用这10组数据来产生图像查看训练过程
img_size = 28
onehot = np.eye(10)
temp_z_ = np.random.normal(0, 1, (10, 1, 1, 30))
fixed_z_ = temp_z_
fixed_y_ = np.zeros((10, 1))
for i in range(9):
    fixed_z_ = np.concatenate([fixed_z_, temp_z_], 0)  # 拼接   直接拼在第0维后面
    temp = np.ones((10, 1)) + i
    fixed_y_ = np.concatenate([fixed_y_, temp], 0)  # 0 1 2 ...

fixed_y_ = onehot[fixed_y_.astype(np.int32)].reshape((100, 1, 1, 10))

def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    test_images = sess.run(G_z, {z: fixed_z_, y_label: fixed_y_, isTrain: False})

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (img_size, img_size)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

###################################################  反向传播部分  ###################################################

# training parameters
learningrate = 0.0002
train_epoch = 30
global_step = tf.Variable(0, trainable=False)  # 记录全局的步数
lr = tf.train.exponential_decay(learningrate, global_step, 500, 0.95, staircase=True)  # 学习率衰减

z_dim = 30  # deliGAN

# load MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

# variables : input
x = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 1))  # 全是4维的矩阵   
y_label = tf.placeholder(tf.float32, shape=(None, 1, 1, 10))     # 标签
y_fill = tf.placeholder(tf.float32, shape=(None, img_size, img_size, 10))  # ？ 应该是填满的标签信息
isTrain = tf.placeholder(dtype=tf.bool)

################ deliGAN  #################
#  zinp =  σ * z  +  μ

# z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))     # 随机数
z = tf.placeholder(tf.float32, [batch_size ,1, 1, z_dim], name="z")   # z为30维的数据
zmu = tf.get_variable("generator_zmu", [batch_size,1 ,1 , z_dim],initializer=tf.random_uniform_initializer(-1,1))   # zin  生成均匀分布的 μ
zsig = tf.get_variable("generator_sig", [batch_size,1 ,1 , z_dim],initializer=tf.constant_initializer(0.2))       # 生成0.2的张量  相当于σ = 0.2
zinp = tf.add(zmu,tf.multiply(z,zsig))  # 这里相当于  zinp = μ + σ * z
zinp = z     				# Uncomment this line when training/testing baseline GAN

# networks : generator
G_z = generator(zinp, y_label, isTrain)  

# networks : discriminator
D_real, D_real_logits = discriminator(x, y_fill, isTrain)                 # 用真的图片训练
D_fake, D_fake_logits = discriminator(G_z, y_fill, isTrain, reuse=True)   # 用G的图片训练  reuse=True 为重复使用

# loss for each network  交叉熵损失函数
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))  # real->1  logits 是最后一层的数值
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1]))) # fake->0
D_loss = D_loss_real + D_loss_fake   # 两个loss加起来才是真loss

# 需要加入sig的损失函数  防止sig为1
sigma_loss = tf.reduce_mean(tf.square(zsig-1))/3    # sigma regularizer   sigma的损失函数是对1的均方误差？  其实就是σ的l2正则化

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))  # fake->1

# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
# 训练方法  Adam法
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    # optim = tf.train.AdamOptimizer(lr, beta1=0.5)
    # D_optim = optim.minimize(D_loss, global_step=global_step, var_list=D_vars)
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars, global_step=global_step)
    # sigma_loss 和 gloss 同梯度下降
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss + sigma_loss, var_list=G_vars)


#############################################  训练部分  #############################################

# open session and initialize all variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# MNIST resize and normalization
# train_set = tf.image.resize_images(mnist.train.images, [img_size, img_size]).eval()
# train_set = (train_set - 0.5) / 0.5  # normalization; range: -1 ~ 1
train_set = (mnist.train.images - 0.5) / 0.5  # 标准化  原始数据就是 0 -1 的数 要将它标准化为-1 - 1
train_label = mnist.train.labels

# results save folder
root = 'MNIST_cDCGAN_results/'
model = 'MNIST_cDCGAN_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')
# 记录hist
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

# training-loop
np.random.seed(int(time.time()))
print('training start!')
start_time = time.time()
for epoch in range(train_epoch): # 迭代轮数
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    shuffle_idxs = random.sample(range(0, train_set.shape[0]), train_set.shape[0]) # 随机训练 
    shuffled_set = train_set[shuffle_idxs]  #  打乱顺序
    shuffled_label = train_label[shuffle_idxs]
    for iter in range(shuffled_set.shape[0] // batch_size):
        # update discriminator    
        x_ = shuffled_set[iter*batch_size:(iter+1)*batch_size]  # 切片
        y_label_ = shuffled_label[iter*batch_size:(iter+1)*batch_size].reshape([batch_size, 1, 1, 10])
        y_fill_ = y_label_ * np.ones([batch_size, img_size, img_size, 10])  # 把他全部格式化成标签
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, z_dim))

        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, y_fill: y_fill_, y_label: y_label_, isTrain: True})  # 先训练D

        # update generator
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, z_dim))
        y_ = np.random.randint(0, 9, (batch_size, 1))  # 随机生成label ？
        y_label_ = onehot[y_.astype(np.int32)].reshape([batch_size, 1, 1, 10])
        y_fill_ = y_label_ * np.ones([batch_size, img_size, img_size, 10])
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, y_fill: y_fill_, y_label: y_label_, isTrain: True})  # 再训练G

        errD_fake = D_loss_fake.eval({z: z_, y_label: y_label_, y_fill: y_fill_, isTrain: False})  # 获得损失
        errD_real = D_loss_real.eval({x: x_, y_label: y_label_, y_fill: y_fill_, isTrain: False})
        errG = G_loss.eval({z: z_, y_label: y_label_, y_fill: y_fill_, isTrain: False})

        D_losses.append(errD_fake + errD_real)
        G_losses.append(errG)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    show_result((epoch + 1), save=True, path=fixed_p)
    train_hist['D_losses'].append(np.mean(D_losses))
    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")
with open(root + model + 'train_hist.pkl', 'wb') as f:
    pickle.dump(train_hist, f)

show_train_hist(train_hist, save=True, path=root + model + 'train_hist.png')

images = []
for e in range(train_epoch):
    img_name = root + 'Fixed_results/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'generation_animation.gif', images, fps=5)

sess.close()