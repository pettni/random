import numpy as np
from nn import *

def test_softmax():

	for u in [2.3, 23.4, 0.2]:

		# f(u) = e^u / (e^u + 2)
		x = np.array([u, 0, 0])

		smax = SoftMax((3,), 3)
		smax.beta = np.eye(3)
		smax.alpha = np.zeros(3)

		smax.forward(x)

		grad = np.array([1, 0, 0])

		dL_dx, dL_dalpha, dL_dbeta = smax.backprop(grad)

		np.testing.assert_almost_equal(
			dL_dx[0], 
			2*np.exp(u)/(np.exp(u) + 2)**2
		)

def test_maxpool():

	pool = MaxPool((4,4), 2)

	x = np.array([[1,2,3,4], [2,5,4,0], [0,0,0,0], [10,0,0,2]])

	y = pool.forward(x)

	grad = np.array([[1,2], [3,4]])

	grad = pool.backprop(grad)

	np.testing.assert_almost_equal(
		grad,
		np.array([[0,0,0,2], [0,1,2,0], [0,0,0,0], [3,0,0,4]])
	)

def test_conv():

	conv = Conv((3,3), 3, 2)
	conv.w = np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2], [2, 2, 2]]])

	x = np.array([[1,2,3], [4,5,6], [7,8,9]])

	y = conv.forward(x)

	np.testing.assert_almost_equal(
		y,
		np.array(np.array([[[sum(x.flatten()), 2*sum(x.flatten())]]]))
	)

	grad = np.array([[[1, 1]]])

	grad, dy_dw = conv.backprop(grad)

	np.testing.assert_almost_equal(
		dy_dw,
		np.array([x, x])
	)
