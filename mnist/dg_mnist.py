# This is the code for experiments performed on the MNIST dataset for the DeLiGAN model. Minor adjustments in
# the code as suggested in the comments can be done to test GAN. Corresponding details about these experiments
# can be found in section 5.3 of the paper and the results showing the outputs can be seen in Fig 4.

import tensorflow as tf
import numpy as np
from ops import *
from utils import *
import os
import time
from random import randint
import cv2
import matplotlib.pylab as Plot
import matplotlib.pyplot as plt
import tsne
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib
import numpy as Math
import sys
from tensorflow.contrib.layers import batch_norm

data_dir='./datasets/mnist/'
results_dir='./results/mnist/'

phase_train = tf.placeholder(tf.bool, name = 'phase_train') # 全局变量
  
# 小批量判别器  用于解决模式崩溃的问题   直接计算批量样本的统计特征
def Minibatch_Discriminator(input, num_kernels=100, dim_per_kernel=5, init=False, name='MD'):
    num_inputs=df_dim*4
    theta = tf.get_variable(name+"/theta",[num_inputs, num_kernels, dim_per_kernel], initializer=tf.random_normal_initializer(stddev=0.05))
    log_weight_scale = tf.get_variable(name+"/lws",[num_kernels, dim_per_kernel], initializer=tf.constant_initializer(0.0))
    W = tf.multiply(theta, tf.expand_dims(tf.exp(log_weight_scale)/tf.sqrt(tf.reduce_sum(tf.square(theta),0)),0))
    W = tf.reshape(W,[-1,num_kernels*dim_per_kernel])
    x = input
    x=tf.reshape(x, [batchsize,num_inputs])
    activation = tf.matmul(x, W)
    activation = tf.reshape(activation,[-1,num_kernels,dim_per_kernel])
    abs_dif = tf.multiply(tf.reduce_sum(tf.abs(tf.subtract(tf.expand_dims(activation,3),tf.expand_dims(tf.transpose(activation,[1,2,0]),0))),2),
                                                1-tf.expand_dims(tf.constant(np.eye(batchsize),dtype=np.float32),1))
    f = tf.reduce_sum(tf.exp(-abs_dif),2)/tf.reduce_sum(tf.exp(-abs_dif))
    print((f.get_shape()))
    print((input.get_shape()))
    return tf.concat(axis=1,values=[x, f])

# 两矩阵相乘  x*w + b
def linear(x,output_dim, name="linear"):
    w=tf.get_variable(name+"/w", [x.get_shape()[1], output_dim])
    b=tf.get_variable(name+"/b", [output_dim], initializer=tf.constant_initializer(0.0))
    return tf.matmul(x,w)+b

# fc 全连接层
def fc_batch_norm(x, n_out, phase_train, name='bn'):
        beta = tf.get_variable(name + '/fc_beta', shape=[n_out], initializer=tf.constant_initializer())
        gamma = tf.get_variable(name + '/fc_gamma', shape=[n_out], initializer=tf.random_normal_initializer(1., 0.02))
        batch_mean, batch_var = tf.nn.moments(x, [0], name=name + '/fc_moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)  # 批标准化
        return normed

# 全连接层 ？？
def global_batch_norm(x, n_out, phase_train, name='bn'):
    beta = tf.get_variable(name + '/beta', shape=[n_out], initializer=tf.constant_initializer(0.))
    gamma = tf.get_variable(name + '/gamma', shape=[n_out], initializer=tf.random_normal_initializer(1., 0.02))
    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name=name + '/moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    return normed

# 卷积
def conv(x,  Wx, Wy,inputFeatures, outputFeatures, stridex=1, stridey=1, padding='SAME', transpose=False, name='conv'):
    w = tf.get_variable(name+"/w",[Wx, Wy, inputFeatures, outputFeatures], initializer=tf.truncated_normal_initializer(stddev=0.02)) # 卷积核
    b = tf.get_variable(name+"/b",[outputFeatures], initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(x, w, strides=[1,stridex,stridey,1], padding=padding) + b
    return conv

# 反卷积  转置卷积
def convt(x, outputShape, Wx=3, Wy=3, stridex=1, stridey=1, padding='SAME', transpose=False, name='convt'):
    w = tf.get_variable(name+"/w",[Wx, Wy, outputShape[-1], x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable(name+"/b",[outputShape[-1]], initializer=tf.constant_initializer(0.0))
    convt = tf.nn.conv2d_transpose(x, w, output_shape=outputShape, strides=[1,stridex,stridey,1], padding=padding) +b
    return convt

# 判别器  y_lable 为改好的 ? 10
def discriminator(image, y_label, Reuse=False):
    with tf.variable_scope('disc', reuse=Reuse):
        # image : ? 784
        image = tf.reshape(image, [-1, 28, 28, 1])
        # 把 y 格式化为标签
        y_label = tf.reshape(y_label,  [batchsize, 1, 1, 10])
        y_fill = y_label * np.ones([batchsize, image_size, image_size, 10])
        # 拼在最后一个维度
        # cat0 ? 28 28 11
        cat0 = tf.concat([image, y_fill], 3)

        # 第一层 5*5 步长为2 的卷积层  生成8（df_dim 16？）个特征   应该不够 
        # 维度 50 14 14 16
        h0 = lrelu(conv(cat0, 5, 5, 10+1, df_dim, stridex=2, stridey=2, name='d_h0_conv')) 
        # 第二层 5*5 步长2  卷积层  生成16（32 df_dim*2）个特征  
        # batch_norm 用于在深度神经网络训练过程中使得每一层神经网络的输入保持相同分布
        # 维度 50 7 7 32
        h1 = lrelu( batch_norm(conv(h0, 5, 5, df_dim,df_dim*2,stridex=2,stridey=2,name='d_h1_conv'), decay=0.9, scale=True, updates_collections=None, is_training=phase_train, reuse=Reuse, scope='d_bn1'))
        # 第三层 3*3 2 生成32个特征
        # 维度 50 4 4 64
        h2 = lrelu(batch_norm(conv(h1, 3, 3, df_dim*2, df_dim*4, stridex=2, stridey=2,name='d_h2_conv'), decay=0.9,scale=True, updates_collections=None, is_training=phase_train, reuse=Reuse, scope='d_bn2'))
        # 第四层  最大值池化操作
        # 50 1 1 64
        h3 = tf.nn.max_pool(h2, ksize=[1,4,4,1], strides=[1,1,1,1],padding='VALID')
        # h6 = tf.reshape(h2,[-1, 4*4*df_dim*4])  # 没有用到？？
        # 小批量判别器层
        # 维度 50 128
        h7 = Minibatch_Discriminator(h3, num_kernels=df_dim*4, name = 'd_MD')
        # 全连接层？
        # 维度 50 1
        h8 = dense(tf.reshape(h7, [batchsize, -1]), df_dim*4*2, 1, scope='d_h8_lin')

        return tf.nn.sigmoid(h8), h8   # D_prob, D_logit

# 生成器
def generator(z, y_lable):
    with tf.variable_scope('gen'):
        # y 变形
        # y = y_lable * np.ones([batchsize, 1, 1, 10])
        cat0 = tf.concat([z, y_lable], 1)   # 在第三个维度拼接

        # 第一层  FC 全连接
        # 维度 ？ 4 4 64
        h0 = tf.reshape(tf.nn.relu(fc_batch_norm(linear(cat0, gf_dim*4*4*4, name='g_h0'), gf_dim*4*4*4, phase_train, 'g_bn0')), [-1, 4, 4, gf_dim*4])
        # 第二层 卷积  3*3 32 
        # 维度 50 7 7 32
        h1 = tf.nn.relu(global_batch_norm(convt(h0,[batchsize, 7, 7, gf_dim*2],3, 3, 2, 2, name='g_h1'), gf_dim*2, phase_train, 'g_bn1'))
        # 第三层 卷积 5*5 16
        # 维度 50 14 14 16
        h3 = tf.nn.relu(global_batch_norm(convt(h1,[batchsize, 14, 14,gf_dim],5, 5, 2, 2, name='g_h3'), gf_dim, phase_train, 'g_bn3'))
        # 第四层 卷积 5*5 1
        # 维度 50 28 28 1
        h4 = tf.tanh(convt(h3,[batchsize, 28, 28, 1], 5, 5, 2, 2, name='g_h4'))
        return h4

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

# 这里已经是在训练过程中了  反向传播过程中
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    batchsize = 50        # 每个batch中训练样本的数量
    imageshape = [28*28]
    image_size = 28
    z_dim = 100   # dim == dimension
    gf_dim = 32  # 特征层大小
    df_dim = 32  # 特征层大小
    learningrate = 0.0005
    beta1 = 0.5

    images = tf.placeholder(tf.float32, [batchsize] + imageshape, name="real_images")
    z = tf.placeholder(tf.float32, [None, z_dim], name="z")   # z为30维的数据
    lr1 = tf.placeholder(tf.float32, name="lr") # 学习率占位
    y_label = tf.placeholder(tf.float32, shape=(None, 10))     # 标签

################   DeliGAN    ########################

    # Our Mixture Model modifications  修改
    zin = tf.get_variable("g_z", [batchsize, z_dim],initializer=tf.random_uniform_initializer(-1,1))  # zin  生成均匀分布的 μ
    zsig = tf.get_variable("g_sig", [batchsize, z_dim],initializer=tf.constant_initializer(0.2)) # 生成0.2的张量  相当于σ = 0.2
    inp = tf.add(zin,tf.multiply(z,zsig))  # 这里相当于  zinp = μ + σ * z
    # inp = z     				# Uncomment this line when training/testing baseline GAN

################   DeliGAN    ########################

    G = generator(inp, y_label)
    D_prob, D_logit = discriminator(images, y_label)  # 训练真实图片
    D_fake_prob, D_fake_logit = discriminator(G, y_label, Reuse=True) # 训练生成器图片



# tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
# 主要用作降维或者计算tensor（图像）的平均值。

    # sigmoid交叉熵损失函数  x = logits, z = labels 
    # max(x, 0) - x * z + log(1 + exp(-abs(x)))
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit, labels=tf.ones_like(D_logit)))  # 真  训练接近1
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logit, labels=tf.zeros_like(D_fake_logit)))  # 假图片的损失 训练接近0

    sigma_loss = tf.reduce_mean(tf.square(zsig-1))/3    # sigma regularizer   sigma的损失函数是对1的均方误差？  其实就是σ的l2正则化
    gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logit, labels=tf.ones_like(D_fake_logit)))  # G的损失  将其训练接近1
    dloss = d_loss_real + d_loss_fake  # 真假loss相加

    t_vars = tf.trainable_variables()     #  所有可训练的参数  
    d_vars = [var for var in t_vars if 'd_' in var.name]  #  D的可训练参数
    g_vars = [var for var in t_vars if 'g_' in var.name]  #  G的可训练参数

    # 读取数据
    data = np.load(data_dir + 'mnist.npz')
    trainx = np.concatenate([data['trainInps']], axis=0)  # 数据
    trainy = np.concatenate([data['trainTargs']], axis=0) # tags
    trainx = 2*trainx/255.-1  # 标椎化？

    #  截取数据
    data = []
    data_label = []
    data_size = 500
    # Uniformly sampling 50 images per category from the dataset  从数据集中为每个类别统一采样50张图像
    for i in range(10):
        train = trainx[np.argmax(trainy,1)==i]
        label = trainy[np.argmax(trainy,1)==i]
        # print(np.argmax(trainy,1)==i)
        data.append(train[-data_size:])
        data_label.append(label[-data_size:])

    data = np.array(data).reshape([-1, image_size*image_size])
    data_label = np.array(data_label).reshape([-1, 10])
    # data = np.reshape(data,[-1,28*28])
    # data_label = np.reshape(data_label,[-1,10])

    # 训练  此函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。  beta1 一阶矩估计的指数衰减率  类似于梯度下降法
    d_optim = tf.train.AdamOptimizer(lr1, beta1=beta1).minimize(dloss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(lr1, beta1=beta1).minimize(gloss + sigma_loss, var_list=g_vars)  # 把sigma引入G的梯度来训练

    tf.global_variables_initializer().run()  # run！！！

    saver = tf.train.Saver(max_to_keep=10)

    onehot = np.eye(10)
    # 生成0-9的标签
    fixed_y = np.zeros((5, 1))
    for i in range(9):
        temp = np.ones((5, 1)) + i
        fixed_y = np.concatenate([fixed_y, temp], 0)  # 0 1 2 ...
    fixed_y = onehot[fixed_y.astype(np.int32)].reshape((batchsize, 10))


    counter = 1
    start_time = time.time()
    data_size = data.shape[0]  # 只取样了50*10 张图片用于训练
    display_z = np.random.uniform(-1.0, 1.0, [batchsize, z_dim]).astype(np.float32)   # 均匀分布

    seed = 1
    rng = np.random.RandomState(seed)
    train = True
    thres=1.0      # used to balance gan training  ？？
    count1=0
    count2=0
    t1=0.70

    if train:
        # saver.restore(sess, tf.train.latest_checkpoint(os.getcwd()+"../results/mnist/train/"))
        # training a model
        for epoch in range(4000):
            batch_idx = data_size//batchsize  # 训练轮数
            radoms = rng.permutation(data_size)
            radom_data = data[radoms]
            radom_labels = data_label[radoms]
            lr = learningrate * (np.minimum((4 - epoch/1000.), 3.)/3)  # 学习率衰减

            # 单批次训练
            for idx in range(batch_idx):
                batch_images = radom_data[idx*batchsize:(idx+1)*batchsize]  # 单批次训练的图像
                batch_labels = radom_labels[idx*batchsize:(idx+1)*batchsize] 

                # uniform 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.
                # batch_z = np.random.uniform(-1.0, 1.0, [batchsize, z_dim]).astype(np.float32) # z是一个（50, 30) 的元组  uniform 是均匀分布  意义？？

                # 用来平衡D和G   作者的方法
                if count1>3:
                    thres=min(thres+0.003, 1.0)
                    count1=0
                    print(("[%2d] gen, %f" % (counter, thres)))
                if count2<-1:
                    thres=max(thres-0.003, t1)
                    count2=0
                    print(("[%2d] disc, %f" % (counter, thres)))

                for k in range(5):  # 每一次betch 有50
                    batch_z = np.random.normal(0, 1.0, [batchsize, z_dim]).astype(np.float32)  # 产生正态分布
                    y_ = np.random.randint(0, 9, (batchsize, 1))  # 随机生成label 
                    y_random = onehot[y_.astype(np.int32)].reshape([batchsize, 10])
                    # 当 gloss > thres 时训练G  
                    if gloss.eval({z: batch_z, y_label:y_random, phase_train.name:False})>thres:    # 在SESSION中评估次张量   这个就是gloss的值
                        sess.run([g_optim],feed_dict={z: batch_z, y_label:y_random , lr1:lr, phase_train.name:True})   # 训练G
                        count1+=1
                        count2=0
                    else:
                        sess.run([d_optim],feed_dict={ images: batch_images, y_label:batch_labels, z: batch_z, lr1:lr, phase_train.name:True})  # 训练D   先训练D
                        count2-=1
                        count1=0
                counter += 1

                if counter % 300 == 0:
                    # Saving 49 randomly generated samples
                    print(("Epoch: [%2d] [%4d/%4d] time: %4.4f, "  % (epoch, idx, batch_idx, time.time() - start_time,)))
                    
                    sdata = sess.run(G,feed_dict={ z: batch_z, y_label:fixed_y, phase_train.name:False})
                    sdata = sdata.reshape(sdata.shape[0], 28, 28, 1)/2.+0.5
                    sdata = merge(sdata[:],[10,5])  # 变为5*10
                    sdata = np.array(sdata*255.,dtype=np.int) # 回复图片数据
                    cv2.imwrite(results_dir + "/" + str(counter) + ".png", sdata)
                    errD_fake = d_loss_fake.eval({z: display_z, y_label:fixed_y, phase_train.name:False})  
                    errD_real = d_loss_real.eval({images: batch_images, y_label:fixed_y, phase_train.name:False})
                    errG = gloss.eval({z: display_z, y_label:fixed_y, phase_train.name:False})
                    sigloss = sigma_loss.eval()
                    print(('D_real: ', errD_real))
                    print(('D_fake: ', errD_fake))
                    print(('G_err: ', errG))
                    print(('sigloss: ', sigloss))
                if counter % 2000 == 0:
                    # Calculating the Nearest Neighbours corresponding to the generated samples  计算与生成的样本相对应的最近邻居
                    sdata = sess.run(G,feed_dict={ z: display_z, y_label:fixed_y, phase_train.name:False})  # 相当于是用均匀分布来产生图片 ？
                    sdata = sdata.reshape(sdata.shape[0], 28*28)
                    NNdiff = np.sum(np.square(np.expand_dims(sdata,axis=1) - np.expand_dims(data,axis=0)),axis=2)  # 找不同
                    NN = data[np.argmin(NNdiff,axis=1)]  # 找到最相似的
                    sdata = sdata.reshape(sdata.shape[0], 28, 28, 1)/2.+0.5
                    NN = np.reshape(NN, [batchsize, 28, 28, 1])/2.+0.5
                    sdata = merge(sdata[:49],[7,7])
                    NN = merge(NN[:49],[7,7])
                    sdata = np.concatenate([sdata, NN], axis=1)
                    sdata = np.array(sdata*255.,dtype=np.int)
                    cv2.imwrite(results_dir + "/NN" + str(counter) + ".png", sdata)#gan_1nin_8gfdim_floss_alpha1_z15

                    # Plotting the latent space using tsne
                    z_Mog = zin.eval()#display_z
                    gen = G.eval({z:display_z, y_label:fixed_y,  phase_train.name:False})
                    Y = tsne.tsne(z_Mog, 2, z_dim, 10.0) #  数据降维 
                    Plot.scatter(Y[:,0], Y[:,1])
                    xtrain = gen.copy()
                    fig, ax = Plot.subplots()
                    artists = []
                    for i, (x0, y0) in enumerate(zip(Y[:,0], Y[:,1])):
                        image = xtrain[i%xtrain.shape[0]]
                        image = image.reshape(28,28)
                        im = OffsetImage(image, zoom=1.0)
                        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
                        artists.append(ax.add_artist(ab))
                    ax.update_datalim(np.column_stack([Y[:,0], Y[:,1]]))
                    ax.autoscale()
                    Plot.scatter(Y[:,0], Y[:,1], 20);
                    fig.savefig(results_dir + "/plot" + str(counter) + ".png")
                    saver.save(sess,"./results/mnist/train/", global_step=counter)
    else:
        #Generating samples from a saved model
        saver.restore(sess,tf.train.latest_checkpoint("./results/mnist/train/"))
        samples=[]
        for i in range(100):
            batch_z = np.random.uniform(-1, 1, [batchsize, z_dim]).astype(np.float32)
            sdata = sess.run(G,feed_dict={z: batch_z, phase_train.name:False})
            sdata = sdata.reshape(sdata.shape[0], 28, 28, 1)/2.+0.5
            sdata = sdata*255.
            samples.append(sdata)
        samples1 = np.concatenate(samples,0)
        np.save(results_dir + '/MNIST_samples5k.npy',samples1)
        print("samples saved")
        sys.exit()

