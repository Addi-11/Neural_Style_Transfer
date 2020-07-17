import tensorflow as tf
import numpy as np
import scipy.io
import scipy.misc
from utils import CONFIG


class VGG19:

	vgg = scipy.io.loadmat(CONFIG.PRE_TRAINED_PATH)
	vgg_layers = vgg['layers']

	def _weights(self, layer, name):
		# matconvnet: weights are [width, height, in_channels, out_channels]
		# tensorflow: weights are [height, width, in_channels, out_channels]
		wb = self.vgg_layers[0][layer][0][0][2]
		W = wb[0][0]
		b = wb[0][1]
		return W, b

	def _relu(self, conv2d_layer):
		return tf.nn.relu(conv2d_layer)

	def _conv2d(self, prev_layer, layer, layer_name):
		W, b = self._weights(layer,layer_name)
		W = tf.constant(W)
		b = tf.constant(np.reshape(b, (b.size)))
		stride = [1,1,1,1]
		padding = 'SAME'
		
		return tf.nn.conv2d(prev_layer, filters=W, strides=stride, padding=padding) + b

	def _conv2d_relu(self, prev_layer, layer, layer_name):
		return self._relu(self._conv2d(prev_layer, layer, layer_name))

	def _avgpool(self, prev_layer):
		padding = 'SAME'
		stride = [1,2,2,1]
		return tf.nn.avg_pool(prev_layer, ksize=[1,2,2,1], strides=stride, padding=padding)


	def load_vgg_model(self):

		# Constructing a graph model
		# we are doing this to replace the maxpool layers of VGG19 with avg pool layers
		graph = {}
		graph['input'] = tf.Variable(np.zeros((1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)), dtype = 'float32')

		prev_layer = 'input'
		
		# layers to br added in our model
		for layer_num in range(self.vgg_layers.shape[1] - 6):
			layer_name = self.vgg_layers[0][layer_num][0][0][0][0]
			layer_type = layer_name[:4]

			if layer_type == 'relu':
				continue

			elif layer_type == 'conv':
				graph[layer_name] = self._conv2d_relu(graph[prev_layer], layer_num, layer_name)

			elif layer_type == 'pool':
				graph['avg'+layer_name] = self._avgpool(graph[prev_layer])
				layer_name = 'avg'+layer_name

			prev_layer = layer_name

		return graph
