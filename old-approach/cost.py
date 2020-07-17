import tensorflow as tf
import numpy as np

class Cost:

	# name of layers whose style we will extract from, adn their co-efficients
	STYLE_LAYERS = [
		('conv1_1', 0.2),
		('conv2_1', 0.2),
		('conv3_1', 0.2),
		('conv4_1', 0.2),
		('conv5_1', 0.2),
	]

	def compute_content_cost(self,aC, aG):
		_, nH, nW, nC = aG.get_shape().as_list()

		normal = 4*nH*nW*nC # normalization constant
		J_content = tf.reduce_sum(tf.square(tf.subtract(aC, aG)))/normal

		return J_content

	def _gram_matrix(self,A):
		return tf.matmul(A, tf.transpose(A))

	def compute_style_layer_cost(self,aS, aG):
		_, nH, nW, nC = aG.get_shape().as_list()
		normal = 4 * (nH*nW)**2 * nC**2 # normalization constant

		# unrolling into 2D matrices
		aS = tf.reshape(tf.transpose(aS), [nC, nH*nW])
		aG = tf.reshape(tf.transpose(aG), [nC, nH*nW])

		# computing gram matrices
		GG = self._gram_matrix(aG)
		GS = self._gram_matrix(aS)

		J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG)))/normal

		return J_style_layer

	def compute_style_cost(self, model):
		J_style = 0

		for layer_name, coeff in self.STYLE_LAYERS:
			out = model[layer_name]

			# setting aS to be the hidden layer activation from the selected layer
			aS = tf.function(out)

			# setting aG to be activations from the same hidden layer number of the genersted image
			aG = out

			# compute style cost for the current layer
			J_style_layer = self.compute_style_layer_cost(aS, aG)

			J_style += coeff * J_style_layer

			return J_style

	def total_cost(self, J_content, J_style, alpha = 10, beta = 40):
		return alpha*J_content + beta*J_style