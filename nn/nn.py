from abc import ABC, abstractmethod
from itertools import product

import numpy as np

class Layer(ABC):

	def __init__(self, input_shape):
		self.input_shape = input_shape

	def __str__(self):
		return "{}: {} --> {}".format(self.name, self.input_shape, self.output_shape)

	@abstractmethod
	def output_shape(self):
		pass

	@abstractmethod
	def forward(self, input):
		pass

	@abstractmethod
	def backprop(self, grad):
		pass


class Conv(Layer):
	'''convolution along last two dimensions'''
	name = "Conv"

	def __init__(self, input_shape, conv_size=3, num_filters=1):

		super().__init__(input_shape)

		if not conv_size%2 == 1:
			raise Exception("size must be odd")

		if not len(input_shape) > 1:
			raise Exception("Conv needs at least 2d input")

		self.conv_size = conv_size
		self.num_filters = num_filters
		self.w = np.random.randn(self.num_filters, self.conv_size, self.conv_size)/conv_size**2;	

	@property
	def output_shape(self):
		out_shape = list(self.input_shape + (self.num_filters,))
		out_shape[-2] -= (self.conv_size-1)
		out_shape[-3] -= (self.conv_size-1)
		return tuple(out_shape)	

	def iter_regions(self, x):
		for i in range(self.output_shape[-3]):
			for j in range(self.output_shape[-2]):
				region = x[..., i: i+self.conv_size, j: j+self.conv_size]
				yield i, j, region

	def forward(self, x):

		if not x.shape == self.input_shape:
			raise Exception("wrong input shape")

		self.x = x
		y = np.zeros(self.output_shape)   # N x n1-nc x n2-nc x nf
		for i,j,region in self.iter_regions(x):
			# region N x nc x nc
			# w      nf x nc x nc
			y[...,i,j,:] = np.sum(region[..., np.newaxis, :, :] * self.w, axis=(-1, -2))   # N x nf

		return y

	def backprop(self, dL_do):

		# dL_do   N x n1-nc x n2-nc x nf

		dL_dw = np.zeros(self.w.shape)       #  nf x nc x nc

		do_dx = np.zeros(self.output_shape + self.input_shape)  # N x n1-nc x n2-nc x nf  x N x n1 x n2
		do_dw = np.zeros(self.output_shape + self.w.shape)      # N x n1-nc x n2-nc x nf  x nf x nc x nc

		for i, j, region in self.iter_regions(self.x):
			dL_do    # N x n1-nc x n2-nc x nf
			region   # N x nc x nc
			dL_dw += np.einsum('...i,...jk->ijk', dL_do[...,i, j, :], region)   # nf x nc x nc

		return None, dL_dw


class MaxPool(Layer):
	'''pooling along first two dimensions'''
	name = "MaxPool"

	def __init__(self, input_shape, pool_size):

		super().__init__(input_shape)
		self.pool_size = pool_size;

	@property
	def output_shape(self):
		out_shape = list(self.input_shape)
		out_shape[0] = (out_shape[0]+self.pool_size-1)//self.pool_size
		out_shape[1] = (out_shape[1]+self.pool_size-1)//self.pool_size
		return tuple(out_shape)

	def forward(self, x):

		if not x.shape == self.input_shape:
			raise Exception("wrong input shape")

		self.y = np.zeros(self.output_shape)
		self.x = x

		for i in range(self.output_shape[0]):
			for j in range(self.output_shape[1]):
				min_i = self.pool_size*i
				max_i = min(self.input_shape[0], self.pool_size*(i+1))
				min_j = self.pool_size*j
				max_j = min(self.input_shape[1], self.pool_size*(j+1))
				self.y[i, j] = np.amax(self.x[min_i:max_i, min_j:max_j], axis=(0, 1))

		return self.y

	def backprop(self, dL_do):
		dL_dx = np.zeros((self.input_shape))

		for in_midx in product(*[range(x) for x in self.input_shape]):
			out_midx = tuple(in_midx[i]//self.pool_size if i < 2 else in_midx[i]
										   for i in range(len(in_midx)))

			if self.y[out_midx] == self.x[in_midx]:
				dL_dx[in_midx] = dL_do[out_midx]

		return dL_dx


class SoftMax(Layer):
	'''dense linear trans + softmax'''

	name = "SoftMax"

	def __init__(self, input_shape, nodes):

		super().__init__(input_shape)
		self.beta = np.random.randn(nodes, np.prod(input_shape)) / np.prod(input_shape)
		self.alpha = np.zeros(nodes)

	@property
	def output_shape(self):
		return len(self.alpha)

	def forward(self, x):

		# y = alpha + beta * x
		# z = e^y
		# o_i = z_i / sum_k(z_k)

		if not x.shape == self.input_shape:
			raise Exception("wrong input shape")

		self.x = x.flatten()
		y = self.alpha + self.beta @ self.x
		z = np.exp(y) 
		self.o = z / np.sum(z)

		return self.o

	def backprop(self, dL_do):

		n_o = self.output_shape
		n_x = np.prod(self.input_shape)

		do_dy = np.diag(self.o) - np.outer(self.o, self.o)       # n_o x n_o

		dy_dx = self.beta                                            # n_o x n_x
		dy_dalpha = np.eye(self.output_shape)                        # n_o x n_o
		dy_dbeta = np.einsum('ij,k->ijk', np.eye(n_o), self.x)       # n_o x n_o x n_x

		dL_dx = dL_do @ do_dy @ dy_dx                     # p x n_x
		dL_dalpha = dL_do @ do_dy @ dy_dalpha             # p x n_o
		dL_dbeta = dL_do @ do_dy @ dy_dbeta               # p x n_o x n_x

		dL_dx = dL_dx.reshape(self.input_shape)

		return dL_dx, dL_dalpha, dL_dbeta