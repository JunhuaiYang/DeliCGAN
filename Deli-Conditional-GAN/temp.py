zmu = tf.get_variable("gen_mu", [
                      batch_size, z_dim], initializer=tf.random_uniform_initializer(-1, 1))
zsig = tf.get_variable("gen_sig", [
                       batch_size, z_dim], initializer=tf.constant_initializer(0.2))
zinput = tf.add(zmu, tf.multiply(z, zsig))

fake_images = gern_net(zinput, y)
D_pro, D_logits = dis_net(images, y, False)
G_pro, G_logits = dis_net(fake_images, y, True)


D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(G_pro), logits=G_logits))
D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_pro), logits=D_logits))
G_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(G_pro), logits=G_logits))
sigma_loss = tf.reduce_mean(tf.square(zsig - 1)) / 3  # sigma L2 regularizer
D_loss = D_real_loss + D_fake_loss
G_loss = G_fake_loss + sigma_loss  # 加上sigma的L2 正则化

opti_D = tf.train.AdamOptimizer(learning_rate=learn_rate, beta1=0.5).minimize(D_loss, var_list=d_var)
opti_G = tf.train.AdamOptimizer(learning_rate=learn_rate, beta1=0.5).minimize(G_loss, var_list=g_var)