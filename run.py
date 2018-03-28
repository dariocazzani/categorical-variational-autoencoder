import tensorflow as tf
import numpy as np
import os
from network import Network

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

model_path = "saved_models/model.ckpt"

sess = tf.InteractiveSession()
network = Network()
train_op = tf.train.AdamOptimizer(learning_rate=network.lr).minimize(network.loss)
tf.global_variables_initializer().run()
saver = tf.train.Saver()

try:
    saver.restore(sess, model_path)
    print("Model restored from file: {}".format(model_path))
except:
    print("Could not restore saved model")

tau0=1.0 # initial temperature
np_temp=tau0
np_lr=0.001
ANNEAL_RATE=0.00003
MIN_TEMP=0.5

step = 0
try:
    while True:
        batch_x, batch_y = mnist.train.next_batch(128)

        _, loss_value = sess.run([train_op, network.loss], feed_dict={
                            network.x: batch_x,
                            network.is_training: True,
                            network.tau: np_temp,
                            network.lr: np_lr})

        if np.isnan(loss_value):
            raise ValueError('Loss value is NaN')

        if step % 500 == 0:
            batch_x, batch_y = mnist.validation.next_batch(128)
            loss_validation = sess.run(network.loss, feed_dict={
                            network.x: batch_x,
                            network.is_training: False})

            print ('step {}: training loss {:.3f} validation loss {:.3f}'.format(step, loss_value, loss_validation))

        if step % 1000 == 0:
            np_temp=np.maximum(tau0 * np.exp(-ANNEAL_RATE * step), MIN_TEMP)
            np_lr*=0.9

        step+=1

except (KeyboardInterrupt, SystemExit):
    print("Manual Interrupt")
    save_path = saver.save(sess, model_path)

except Exception as e:
    print("Exception: {}".format(e))
