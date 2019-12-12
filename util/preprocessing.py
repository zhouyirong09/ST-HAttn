import numpy as np
import copy
import scipy.stats as stats


class BoxCox_MinMax(object):
	def __init__(self, l_normal=''):
		pass

	def fit(self, data):
		dx = data.flatten()
		dx = dx[dx != 0]
		dtmp, self.l = stats.boxcox(dx+1)
		self._min = np.amin(dtmp)
		self._max = np.amax(dtmp)
		print("min: ", self._min, "max:", self._max)

	def get_max(self):
		return np.power((self._max * self.l + 1), 1/self.l) - 1

	def transform(self, data):
		dtmp = (np.power(data+1, self.l)-1) / self.l
		norm_data = 1. * (dtmp - self._min) / (self._max - self._min)
		return norm_data

	def fit_transform(self, data):
		self.fit(data)
		return self.transform(data)

	def inverse_transform(self, data):
		inverse_norm_data = 1. * data * (self._max - self._min) + self._min
		inverse_norm_data = np.power((inverse_norm_data * self.l + 1), 1/self.l) - 1
		return inverse_norm_data

	def real_loss(self, loss):
		# loss is rmse
		return np.power((loss * self.l + 1), 1/self.l) - 1
		#return real_loss



class LogMinMax(object):
	def __init__(self, l_normal=''):
		pass

	def fit(self, data):
		dtmp = np.log(data+1)
		self._min = np.amin(dtmp)
		self._max = np.amax(dtmp)
		print("min: ", self._min, "max:", self._max)

	def get_max(self):
		return np.exp(self._max)-1

	def transform(self, data):
		dtmp = np.log(data+1)
		norm_data = 1. * (dtmp - self._min) / (self._max - self._min)
		return norm_data

	def fit_transform(self, data):
		self.fit(data)
		return self.transform(data)

	def inverse_transform(self, data):
		inverse_norm_data = 1. * data * (self._max - self._min) + self._min
		inverse_norm_data = np.exp(inverse_norm_data)-1
		return inverse_norm_data

	def real_loss(self, loss):
		# loss is rmse
		return np.exp(loss*(self._max - self._min))-1
		#return real_loss


class MinMaxNormalization01(object):
	def __init__(self, l_normal='0'):
		pass

	def fit(self, data):
		self._min = np.amin(data)
		self._max = np.amax(data)
		print("min: ", self._min, "max:", self._max)

	def get_max(self):
		return self._max

	def transform(self, data):
		norm_data = 1. * (data - self._min) / (self._max - self._min)
		return norm_data

	def fit_transform(self, data):
		self.fit(data)
		return self.transform(data)

	def inverse_transform(self, data):
		inverse_norm_data = 1. * data * (self._max - self._min) + self._min
		return inverse_norm_data

	def real_loss(self, loss):
		# loss is rmse
		return loss*(self._max - self._min)
		#return real_loss


class SameTrans(object):
	def __init__(self, ):
		pass

	def fit(self, data):
		self._min = np.amin(data)
		self._max = np.amax(data)
		print("SameTrans min: ", self._min, "max:", self._max)

	def get_max(self):
		return self._max

	def transform(self, data):
		return data

	def fit_transform(self, data):
		return data

	def inverse_transform(self, data):
		return data

	def real_loss(self, loss):
		# loss is rmse
		return loss
		#return real_loss

class PowerTrans(object):
	def __init__(self, l_normal='0'):
		pass

	def fit(self, data):
		self._min = np.amin(data)
		self._max = np.amax(data)
		print("PowerTrans min: ", self._min, "max:", self._max)

	def get_max(self):
		return self._max

	def transform(self, data):
		norm_data = 1. * (data - self._min) / (self._max - self._min)
		norm_data = np.power(norm_data, 2)
		return norm_data

	def fit_transform(self, data):
		self.fit(data)
		return self.transform(data)

	def inverse_transform(self, data):
		data = np.sqrt(data)
		inverse_norm_data = 1. * data * (self._max - self._min) + self._min
		return inverse_norm_data

	def real_loss(self, loss):
		# loss is rmse
		return np.sqrt(loss)*(self._max - self._min)
		#return real_loss

class CurveTrans(object):
	def __init__(self, ):
		pass

	def fit(self, data):
		self._min = np.amin(data)
		self._max = np.amax(data)
		print("CurveTrans min: ", self._min, "max:", self._max)

	def get_max(self):
		return self._max

	def transform(self, data):
		norm_data = 1. * (data - self._min) / (self._max - self._min)
		norm_data = np.sqrt(norm_data)+0.1
		return norm_data

	def fit_transform(self, data):
		self.fit(data)
		return self.transform(data)

	def inverse_transform(self, data):
		data = np.power(data-0.1, 2)
		inverse_norm_data = 1. * data * (self._max - self._min) + self._min
		return inverse_norm_data

	def real_loss(self, loss):
		# loss is rmse
		return loss*loss*(self._max - self._min)
		#return real_loss

class Histogram_Normalize(object):
	def __init__(self, l_normal='0'):
		self.dj = {}
		self.l_normal = l_normal[-1]
		pass

	def get_max(self):
		return self._max
	def cdf_line(self, data):
		a_min, a_max = float(self.cdf.min()), float(self.cdf.max())
		x_min, x_max = float(self._min), float(self._max)
		k = (x_max - x_min) / (a_max - a_min)
		b = x_min - k * a_min
		return k * data + b
	def cdf_rline(self, data):
		x_min, x_max = float(self.cdf.min()), float(self.cdf.max())
		a_min, a_max = float(self._min), float(self._max)
		k = (x_max - x_min) / (a_max - a_min)
		b = x_max - k * a_max
		return k * data + b

	def fit(self, data):
		data = data.astype('int')
		self._max, self._min = data.max(), data.min()
		if self.l_normal == '1':
			w_data = np.bincount(data.flatten())
			w_data = np.log(w_data+1)
			# wk = 10 / w_data.max()
			# w_data = wk * w_data + np.log(2)
			wk = np.median(w_data)
			w_data = 1 / (1 + np.exp((-w_data+wk)))
			histogram, self.bins = np.histogram(np.array(range(self._max+1)), int(self._max)+1, density=True, weights=np.log(w_data+1)+0.1)
		elif self.l_normal == '2':
			w_data = np.bincount(data.flatten())
			w_data[0] = 0
			w_data = w_data + 1
			histogram, self.bins = np.histogram(np.array(range(self._max+1)), int(self._max)+1, density=True, weights=w_data/w_data.sum())
		else:
			histogram, self.bins = np.histogram(data.flatten(), int(self._max)+1, density=True)
		self.cdf = histogram.cumsum()
		self.cdf = 1.0 * self.cdf / self.cdf[-1]

	def transform(self, data):
		equalized = np.interp(data.flatten(), self.bins[:-1], self.cdf)
		return equalized.reshape(data.shape)

	def inverse_transform(self, data):
		reequalized = np.interp(data.flatten(), self.cdf, self.bins[:-1])
		return reequalized.reshape(data.shape)

	def real_loss(self, loss):
		# loss is rmse
		return loss*(self._max - self._min)

class Histogram_Normalize1(object):
	def __init__(self):
		self.dj = {}
		pass

	def get_max(self):
		return self._max

	def fit(self, X):
		Y = copy.copy(X)
		self._min = np.amin(X)
		self._max = np.amax(X)
		print("min: ", self._min, "max:", self._max)
		Y = Y.reshape(-1)
		Y.sort()

		yshape = Y.shape[0]
		for i in range(yshape):
			self.dj[int(Y[i])] = i
		for k in self.dj.keys():
			self.dj[k] /= yshape
		self.keys = list(self.dj.keys())
		self.values = list(self.dj.values())
		self.tk = [-1]
		self.kt = [-1]
		for i in range(len(self.keys)-1):
			self.tk.append((self.values[i] - self.values[i+1]) / (self.keys[i] - self.keys[i+1]))
			self.kt.append((self.keys[i] - self.keys[i+1]) / (self.values[i] - self.values[i+1]))

	def transform(self, data):
		xshape = data.shape
		Y = copy.copy(data.reshape(-1))
		yshape = Y.shape[0]
		for i in range(yshape):
			if int(Y[i]) in self.dj.keys():
				Y[i] = self.dj[int(Y[i])]
			else:
				idx = np.searchsorted(self.keys, Y[i])
				if idx == len(self.keys):
					idx1 = idx - 2
					idx2 = idx - 1
				else:
					idx1 = idx - 1
					idx2 = idx
				Y[i] = self.values[idx1] + self.tk[idx2] * (Y[i] - self.keys[idx1])

		return Y.reshape(xshape)

	def inverse_transform(self, data):
		xshape = data.shape
		Y = copy.copy(data.reshape(-1))
		yshape = Y.shape[0]
		for i in range(yshape):
			idx = np.searchsorted(self.values, Y[i])
			if idx == 0:
				Y[i] = 0
				continue
			if idx == len(self.values):
				idx1 = idx - 2
				idx2 = idx - 1
			else:
				idx1 = idx - 1
				idx2 = idx
			Y[i] = self.keys[idx1] + self.kt[idx2] * (Y[i] - self.values[idx1])

		return Y.reshape(xshape)

	def real_loss(self, loss):
		# loss is rmse
		return loss*(self._max - self._min)



class MinMaxNormalization_neg_1_pos_1(object):
	def __init__(self):
		pass

	def fit(self, X):
		self._min = X.min()
		self._max = X.max()
		print("min:", self._min, "max:", self._max)

	def transform(self, X):
		X = 1. * (X - self._min) / (self._max - self._min)
		X = X * 2. - 1.
		return X

	def fit_transform(self, X):
		self.fit(X)
		return self.transform(X)

	def inverse_transform(self, X):
		X = (X + 1.)/2.
		X = 1. * X * (self._max - self._min) + self._min
		return X

	def real_loss(self, loss):
		# loss is rmse
		return loss*(self._max - self._min)/2.
		#return real_loss
