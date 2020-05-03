from model_mnist import CGAN
import tensorflow as tf
from utils import Mnist
import os

flags = tf.app.flags

flags.DEFINE_string("sample_dir", "DeliCGAN-results/result_image", "the dir of sample images")  # 保存目录
flags.DEFINE_integer("output_size", 28, "the size of generate image")
flags.DEFINE_float("learn_rate", 0.0002, "the learning rate for gan")
flags.DEFINE_integer("batch_size", 64, "the batch number")
flags.DEFINE_integer("epoch", 8000, "the epoch number")
flags.DEFINE_integer("z_dim", 100, "the dimension of noise z")
flags.DEFINE_integer("y_dim", 10, "the dimension of condition y")
flags.DEFINE_string("log_dirs", "DeliCGAN-results/tmp/tensorflow_mnist", "the path of tensorflow's log")
flags.DEFINE_string("model_path", "DeliCGAN-results/model/model.ckpt", "the path of model")
flags.DEFINE_string("visua_path", "DeliCGAN-results/visualization", "the path of visuzation images")  # 显示 
flags.DEFINE_integer("op", 3, "0: train ; 1:test ; 2:visualize ; 3:generate")
flags.DEFINE_integer("generate_number", 500, "the number of generate image epoch")
flags.DEFINE_string("generate_path", "generate-images", "the path of generate images")  # 显示 


FLAGS = flags.FLAGS

if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
if not os.path.exists(FLAGS.log_dirs):
    os.makedirs(FLAGS.log_dirs)
if not os.path.exists(FLAGS.model_path):
    os.makedirs(FLAGS.model_path)
if not os.path.exists(FLAGS.visua_path):
    os.makedirs(FLAGS.visua_path)
if not os.path.exists(FLAGS.generate_path):
    os.makedirs(FLAGS.generate_path)


def main(_):
    mn_object = Mnist()

    cg = CGAN(data_ob=mn_object, sample_dir=FLAGS.sample_dir, output_size=FLAGS.output_size, learn_rate=FLAGS.learn_rate, batch_size=FLAGS.batch_size,
              z_dim=FLAGS.z_dim, y_dim=FLAGS.y_dim, log_dir=FLAGS.log_dirs, model_path=FLAGS.model_path, visua_path=FLAGS.visua_path, epoch=FLAGS.epoch,
              generate_number=FLAGS.generate_number, generate_path=FLAGS.generate_path )

    cg.build_model()

    if FLAGS.op == 0:
        cg.train()
    elif FLAGS.op == 1:
        cg.test()
    elif FLAGS.op == 2:
        cg.visual()
    elif FLAGS.op == 3:
        cg.generate_image()
        


if __name__ == '__main__':
    tf.app.run()
