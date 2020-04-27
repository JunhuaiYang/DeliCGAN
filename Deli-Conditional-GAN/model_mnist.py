from utils import save_images, vis_square,sample_label
from tensorflow.contrib.layers.python.layers import xavier_initializer
import cv2
from ops import conv2d, lrelu, de_conv, fully_connect, conv_cond_concat, batch_normal
import tensorflow as tf
import numpy as np

class CGAN(object):

    # build model
    def __init__(self, data_ob, sample_dir, output_size, learn_rate, batch_size, z_dim, y_dim, log_dir
         , model_path, visua_path):

        self.data_ob = data_ob      # 数据集对象
        self.sample_dir = sample_dir  
        self.output_size = output_size
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.log_dir = log_dir
        self.model_path = model_path
        self.vi_path = visua_path
        self.channel = self.data_ob.shape[2]   # channel=1, 黑白
        self.images = tf.placeholder(tf.float32, [batch_size, self.output_size, self.output_size, self.channel])  
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim])
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim])

    def build_model(self):

        ####################   DeliGAN   #################################
        #  zinp =  σ * z  +  μ
        #  Our Mixture Model modifications，点乘，构造高斯混合模型
        zmu = tf.get_variable("g_z", [self.batch_size, self.z_dim],    initializer=tf.random_uniform_initializer(-1, 1))
        zsig = tf.get_variable("g_sig", [self.batch_size, self.z_dim], initializer=tf.constant_initializer(0.2))
        zinput = tf.add(zmu, tf.multiply(self.z, zsig))
        ####################   DeliGAN   #############################

        #self.fake_images = self.gern_net(self.z, self.y)
        #构建的图像的Tensor必须是4-D形状[batch_size, height, width, channels], 其中channels可以是
        #1： tensor被解析成灰度图像;3：tensor被解析成RGB图像;4：tensor被解析成RGBA图像

        # zinput = self.z   # baselin GAN
        self.fake_images = self.gern_net(zinput, self.y)

        G_image = tf.summary.image("G_out", self.fake_images)  # 输出Summary带有图像的协议缓冲区(name,tensor) 

        ##the loss of gerenate network
        D_pro, D_logits = self.dis_net(self.images, self.y, False)
        D_pro_sum = tf.summary.histogram("D_pro", D_pro) #查看一个张量在训练过程中值的分布情况时，可通过tf.summary.histogram()将其分布情况以直方图的形式在TensorBoard直方图仪表板上显示

        G_pro, G_logits = self.dis_net(self.fake_images, self.y, True)
        G_pro_sum = tf.summary.histogram("G_pro", G_pro)

        D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(G_pro), logits=G_logits))

        D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_pro), logits=D_logits))
        G_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(G_pro), logits=G_logits))

        sigma_loss = tf.reduce_mean(tf.square(zsig - 1)) / 3  # sigma L2 regularizer

        self.D_loss = D_real_loss + D_fake_loss
        self.G_loss = G_fake_loss + sigma_loss  #改动加上sigma的L2 正则化
        #self.G_loss = G_fake_loss

        loss_sum = tf.summary.scalar("D_loss", self.D_loss)  #一般在画loss,accuary时会用到这个函数
        G_loss_sum = tf.summary.scalar("G_loss", self.G_loss)

        self.merged_summary_op_d = tf.summary.merge([loss_sum, D_pro_sum])
        self.merged_summary_op_g = tf.summary.merge([G_loss_sum, G_pro_sum, G_image])

        t_vars = tf.trainable_variables()
        self.d_var = [var for var in t_vars if 'dis' in var.name]
        self.g_var = [var for var in t_vars if 'gen' in var.name]  ## ???

        self.saver = tf.train.Saver()

    def train(self):

        opti_D = tf.train.AdamOptimizer(learning_rate=self.learn_rate, beta1=0.5).minimize(self.D_loss, var_list=self.d_var)
        opti_G = tf.train.AdamOptimizer(learning_rate=self.learn_rate, beta1=0.5).minimize(self.G_loss, var_list=self.g_var)
        init = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:

            sess.run(init)
            summary_writer = tf.summary.FileWriter(self.log_dir, graph=sess.graph)
            #指定一个文件用来保存图，可以调用其add_summary（）方法将训练过程数据保存在filewriter指定的文件中

            step = 0
            while step <= 10000:

                realbatch_array, real_labels = self.data_ob.getNext_batch(step)

                # Get the z
                #batch_z = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim])  #初始版本均匀分布,我们try混合高斯
                ### batch_z = np.random.normal(0 , 0.2 , size=[batch_size , sample_size])   #初始代码就被注释了
                batch_z = np.random.normal(0, 1.0, [self.batch_size, self.z_dim]).astype(np.float32)  # 正态

                _, summary_str = sess.run([opti_D, self.merged_summary_op_d],
                                          feed_dict={self.images: realbatch_array, self.z: batch_z, self.y: real_labels})
                summary_writer.add_summary(summary_str, step)

                _, summary_str = sess.run([opti_G, self.merged_summary_op_g],
                                          feed_dict={self.z: batch_z, self.y: real_labels})
                summary_writer.add_summary(summary_str, step)

                if step % 50 == 0:

                    D_loss = sess.run(self.D_loss, feed_dict={self.images: realbatch_array, self.z: batch_z, self.y: real_labels})
                    fake_loss = sess.run(self.G_loss, feed_dict={self.z: batch_z, self.y: real_labels})
                    print("Step %d: D: loss = %.7f G: loss=%.7f " % (step, D_loss, fake_loss))

                if np.mod(step, 50) == 1 and step != 0:

                    sample_images = sess.run(self.fake_images, feed_dict={self.z: batch_z, self.y: sample_label()})
                    save_images(sample_images, [8, 8],
                                './{}/train_{:04d}.png'.format(self.sample_dir, step))

                    self.saver.save(sess, self.model_path)

                step = step + 1

            save_path = self.saver.save(sess, self.model_path)
            print ("Model saved in file: %s" % save_path)

    def test(self):

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            self.saver.restore(sess, self.model_path)
            sample_z = np.random.uniform(1, -1, size=[self.batch_size, self.z_dim])

            output = sess.run(self.fake_images, feed_dict={self.z: sample_z, self.y: sample_label()})

            save_images(output, [8, 8], './{}/test{:02d}_{:04d}.png'.format(self.sample_dir, 0, 0))

            image = cv2.imread('./{}/test{:02d}_{:04d}.png'.format(self.sample_dir, 0, 0), 0)

            cv2.imshow("test", image)

            cv2.waitKey(-1)

            print("Test finish!")

    def visual(self):

        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)

            self.saver.restore(sess, self.model_path)

            realbatch_array, real_labels = self.data_ob.getNext_batch(0)
            batch_z = np.random.uniform(-1, 1, size=[self.batch_size, self.z_dim])
            # visualize the weights 1 or you can change weights_2 .
            conv_weights = sess.run([tf.get_collection('weight_2')])
            vis_square(self.vi_path, conv_weights[0][0].transpose(3, 0, 1, 2), type=1)

            # visualize the activation 1
            ac = sess.run([tf.get_collection('ac_2')],
                          feed_dict={self.images: realbatch_array[:64], self.z: batch_z, self.y: sample_label()})

            vis_square(self.vi_path, ac[0][0].transpose(3, 1, 2, 0), type=0)

            print("the visualization finish!")

    def gern_net(self, z, y):   #G的输出层不加BN层

        with tf.variable_scope('generator') as scope:

            yb = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])
            z = tf.concat([z, y], 1)  #在同一行的后面加，即列数增加  (64，100+10)
            c1, c2 = int( self.output_size / 4), int(self.output_size / 2 ) #7，14

            # 10 stand for the num of labels
            d1 = tf.nn.relu(batch_normal(fully_connect(z, output_size=1024, scope='gen_fully'), scope='gen_bn1'))  #(64,1024)

            d1 = tf.concat([d1, y], 1)  #(64,1034)

            d2 = tf.nn.relu(batch_normal(fully_connect(d1, output_size=7*7*2*64, scope='gen_fully2'), scope='gen_bn2'))  #c1*c1*2*self.batch_size？？？
            #64,7*7*2*64

            d2 = tf.reshape(d2, [self.batch_size, c1, c1, 64 * 2])  #64,7,7,128
            d2 = conv_cond_concat(d2, yb)#64,7,7,138

            d3 = tf.nn.relu(batch_normal(de_conv(d2, output_shape=[self.batch_size, c2, c2, 128], name='gen_deconv1'), scope='gen_bn3'))#64,14,14,128

            d3 = conv_cond_concat(d3, yb)#64,14,14,138

            d4 = de_conv(d3, output_shape=[self.batch_size, self.output_size, self.output_size, self.channel], 
                         name='gen_deconv2', initializer = xavier_initializer())#64,28,28,1

            return tf.nn.sigmoid(d4)

    def dis_net(self, images, y, reuse=False):    #D的输入层不加BN层

        with tf.variable_scope("discriminator") as scope:

            if reuse == True:
                scope.reuse_variables()

            # mnist data's shape is (28 , 28 , 1)
            yb = tf.reshape(y, shape=[self.batch_size, 1, 1, self.y_dim])
            # concat
            concat_data = conv_cond_concat(images, yb)

            conv1, w1 = conv2d(concat_data, output_dim=10, name='dis_conv1')
            tf.add_to_collection('weight_1', w1)

            conv1 = lrelu(conv1)
            conv1 = conv_cond_concat(conv1, yb)
            tf.add_to_collection('ac_1', conv1)


            conv2, w2 = conv2d(conv1, output_dim=64, name='dis_conv2')
            tf.add_to_collection('weight_2', w2)

            conv2 = lrelu(batch_normal(conv2, scope='dis_bn1'))
            tf.add_to_collection('ac_2', conv2)  #将元素conv2添加到列表ac_2中

            conv2 = tf.reshape(conv2, [self.batch_size, -1])
            conv2 = tf.concat([conv2, y], 1)

            f1 = lrelu(batch_normal(fully_connect(conv2, output_size=1024, scope='dis_fully1'), scope='dis_bn2', reuse=reuse))
            f1 = tf.concat([f1, y], 1)

            out = fully_connect(f1, output_size=1, scope='dis_fully2',  initializer = xavier_initializer())

            return tf.nn.sigmoid(out), out