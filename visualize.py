import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

from network import Network

model_path = "saved_models/model.ckpt"

sess = tf.InteractiveSession()
network = Network()
tf.global_variables_initializer().run()
saver = tf.train.Saver()

try:
	saver.restore(sess, model_path)
	print("Model restored from file: {}".format(model_path))
except:
	raise ValueError("Could not restore saved model")

def visualize():
	sample_digit, _ = mnist.validation.next_batch(1)
	sample_digit = np.squeeze(sample_digit)

	x_p = network.p_x.mean()
	# Get binary encoding for the selected digit
	binary_encoding = sess.run(network.binary_encoding, feed_dict={
							network.x: sample_digit[None, :],
							network.is_training: False})

	# Get reconstructed image
	reconstructed = sess.run(x_p,{network.y:binary_encoding[None, :, :]})

	binary = np.zeros((network.N, network.K))
	# Get on bits from the binary encoding
	indexes = np.argmax(binary_encoding[:], axis=1)

	reconstructed_list = []
	binary_list = []
	for row in range(network.N):
		for j in range(network.K):
			binary = np.zeros((network.N, network.K))
			for i in range(network.N):
				if i == row:
					binary[i][(indexes[i]+j) % network.K] = 1.0
					binary[i][(indexes[i]) % network.K] = 1.0
				else:
					binary[i][(indexes[i]+0) % network.K] = 1.0
			reconstructed_list.append(sess.run(x_p, {network.y: binary[None, :, :]}))
			binary_list.append(binary)

	# fig = plt.figure(figsize=(8, 6))
	# for idx, rec in enumerate(reconstructed_list):
	# 	plt.subplot(30, 10, idx+1)
	# 	plt.imshow(np.reshape(rec, [28, 28]))
	# 	plt.grid('Off')
	# 	plt.subplots_adjust(hspace=0.0, wspace=0.0)
	# 	plt.gca().axis('off')
	# plt.show()

	fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
	ax = ax.flatten()
	for i in range(25):
		ax[i].imshow(np.reshape(reconstructed_list[i], [28, 28]), cmap='gray')
	ax[0].set_xticks([])
	ax[0].set_yticks([])
	plt.tight_layout()
	plt.show()
if __name__ == '__main__':
	visualize()
