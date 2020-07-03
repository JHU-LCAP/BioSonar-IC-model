import tensorflow as tf
import numpy as np
import h5py
import os, glob
import logging

from utilities import *
from ops import *

# Version description (Version, Date, Description)
# v1.0	July 3rd 2020		initial version

# define parameters
nLayers	      = 3			# depth
trials        = 'T01'		# repetition
avg_act_node  = 10 			# sparsity constraint 1 to 100
slope         = 0.2	 		# nonlinearity
max_epoch     = 1000
batch_size    = 250
learning_rate = 1.0e-4
lambda_reg    = 1.0e-4

initDir	      = './Model/CNN_a%0.1f_%s/p%02d/conv%d/' % (slope, trials, avg_act_node, nLayers-1)
paramsDir     = './Model/CNN_a%0.1f_%s/p%02d/conv%d/' % (slope, trials, avg_act_node, nLayers)
logDir        = './Log/CNN_a%0.1f_%s/' % (slope, trials)

if not os.path.isdir(reconDir):
	os.makedirs(reconDir)

if not os.path.isdir(logDir):
	os.makedirs(logDir)

logPath       = logDir + 'conv%d_p%02d.log' % (nLayers, avg_act_node)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S', filename=logPath, filemode='w')

# load data
nFrq = 128					# The number of frequency bins: following NSL toolbox
nTrnData = 15000			# Total number of training samples
nValData =  1500			# Total number of validation samples
dataPath = './Data/USE_YOUR_DATA.mat'	# Load Auditory Spectrogram
with h5py.File(dataPath) as dat:
	allData = np.transpose(np.array(dat['Data'].value, dtype='float32'), (1,0))
nData, nFeatDim = allData.shape[0], allData.shape[1]
nFrm = nFeatDim/nFrq
allData = DataRegularization(allData)

data_idx = np.arange(nData)
np.random.shuffle(data_idx)
trn_idx = data_idx[:nTrnData]
val_idx = data_idx[nTrnData:nTrnData+nValData]
print('Completed data loading.')

# transforming for convolution
TrnData = np.reshape(allData[trn_idx], [nTrnData, nFrm, nFrq, 1])
TrnData = np.transpose(TrnData, (0, 2, 1, 3))
ValData = np.reshape(allData[val_idx], [nValData, nFrm, nFrq, 1])
ValData = np.transpose(ValData, (0, 2, 1, 3))

# define functions
def loadParams(loadPath, dimension, varname):
	params = np.load(loadPath)
	return tf.get_variable(name=varname, shape=dimension, initializer=tf.constant_initializer(params))

def saveParams(sess, saveDir, msg):
	params = sess.run(model)
	if not os.path.isdir(saveDir):
		os.makedirs(saveDir)
	
	fp=open(saveDir+"stop_log.txt", "w")
	fp.write(msg)
	fp.close()

	np.save(saveDir+'wo.npy', params[0])
	np.save(saveDir+'enc1_f1.npy', params[1])
	np.save(saveDir+'enc1_f2.npy', params[2])
	np.save(saveDir+'enc1_f3.npy', params[3])
	np.save(saveDir+'enc1_f4.npy', params[4])
	np.save(saveDir+'enc1_b1.npy', params[5])
	np.save(saveDir+'enc1_b2.npy', params[6])
	np.save(saveDir+'enc1_b3.npy', params[7])
	np.save(saveDir+'enc1_b4.npy', params[8])

	np.save(saveDir+'enc2_f1.npy', params[9])
	np.save(saveDir+'enc2_f2.npy', params[10])
	np.save(saveDir+'enc2_f3.npy', params[11])
	np.save(saveDir+'enc2_b1.npy', params[12])
	np.save(saveDir+'enc2_b2.npy', params[13])
	np.save(saveDir+'enc2_b3.npy', params[14])

	np.save(saveDir+'enc3_f1.npy', params[15])
	np.save(saveDir+'enc3_f2.npy', params[16])
	np.save(saveDir+'enc3_b1.npy', params[17])
	np.save(saveDir+'enc3_b2.npy', params[18])
	

def extractCodeSample(h):
	h_shape = tf.shape(h)
	disth = tf.distributions.Bernoulli(probs=h, dtype=tf.float32)
	return tf.reshape(disth.sample(1), shape=h_shape)


def encoding(x, loadPath=None):
	if loadPath is None:
		with tf.variable_scope(name_or_scope='enc1') as scope:
			l1_f1 = tf.get_variable(name='kernel1', shape=[3,3, 1, 2], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
			l1_f2 = tf.get_variable(name='kernel2', shape=[5,5, 1, 2], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
			l1_f3 = tf.get_variable(name='kernel3', shape=[7,7, 1, 2], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
			l1_f4 = tf.get_variable(name='kernel4', shape=[9,9, 1, 2], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

			l1_b1 = tf.get_variable(name='bias1', shape=[2], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
			l1_b2 = tf.get_variable(name='bias2', shape=[2], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
			l1_b3 = tf.get_variable(name='bias3', shape=[2], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
			l1_b4 = tf.get_variable(name='bias4', shape=[2], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
	

		with tf.variable_scope(name_or_scope='enc2') as scope:
			l2_f1 = tf.get_variable(name='kernel1', shape=[3,3, 8, 4], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
			l2_f2 = tf.get_variable(name='kernel2', shape=[5,5, 8, 4], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
			l2_f3 = tf.get_variable(name='kernel3', shape=[7,7, 8, 4], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

			l2_b1 = tf.get_variable(name='bias1', shape=[4], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
			l2_b2 = tf.get_variable(name='bias2', shape=[4], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
			l2_b3 = tf.get_variable(name='bias3', shape=[4], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
	else:
		with tf.variable_scope(name_or_scope='enc1') as scope:
			encf1 = np.load(loadPath+'enc1_f1.npy')
			encf2 = np.load(loadPath+'enc1_f2.npy')
			encf3 = np.load(loadPath+'enc1_f3.npy')
			encf4 = np.load(loadPath+'enc1_f4.npy')
			
			l1_f1 = tf.get_variable(name='kernel1', shape=[3,3,1,2], initializer=tf.constant_initializer(encf1))
			l1_f2 = tf.get_variable(name='kernel2', shape=[5,5,1,2], initializer=tf.constant_initializer(encf2))
			l1_f3 = tf.get_variable(name='kernel3', shape=[7,7,1,2], initializer=tf.constant_initializer(encf3))
			l1_f4 = tf.get_variable(name='kernel4', shape=[9,9,1,2], initializer=tf.constant_initializer(encf4))

			encb1 = np.load(loadPath+'enc1_b1.npy')
			encb2 = np.load(loadPath+'enc1_b2.npy')
			encb3 = np.load(loadPath+'enc1_b3.npy')
			encb4 = np.load(loadPath+'enc1_b4.npy')
			
			l1_b1 = tf.get_variable(name='bias1', shape=[2], initializer=tf.constant_initializer(encb1))
			l1_b2 = tf.get_variable(name='bias2', shape=[2], initializer=tf.constant_initializer(encb2))
			l1_b3 = tf.get_variable(name='bias3', shape=[2], initializer=tf.constant_initializer(encb3))
			l1_b4 = tf.get_variable(name='bias4', shape=[2], initializer=tf.constant_initializer(encb4))
	
		with tf.variable_scope(name_or_scope='enc2') as scope:
			encf1 = np.load(loadPath+'enc2_f1.npy')
			encf2 = np.load(loadPath+'enc2_f2.npy')
			encf3 = np.load(loadPath+'enc2_f3.npy')

			l2_f1 = tf.get_variable(name='kernel1', shape=[3,3,8,4], initializer=tf.constant_initializer(encf1))
			l2_f2 = tf.get_variable(name='kernel2', shape=[5,5,8,4], initializer=tf.constant_initializer(encf2))
			l2_f3 = tf.get_variable(name='kernel3', shape=[7,7,8,4], initializer=tf.constant_initializer(encf3))
	
			encb1 = np.load(loadPath+'enc2_b1.npy')
			encb2 = np.load(loadPath+'enc2_b2.npy')
			encb3 = np.load(loadPath+'enc2_b3.npy')

			l2_b1 = tf.get_variable(name='bias1', shape=[4], initializer=tf.constant_initializer(encb1))
			l2_b2 = tf.get_variable(name='bias2', shape=[4], initializer=tf.constant_initializer(encb2))
			l2_b3 = tf.get_variable(name='bias3', shape=[4], initializer=tf.constant_initializer(encb3))

	
	with tf.variable_scope(name_or_scope="E") as scope:
		weights = {
			'ew': tf.get_variable(name='wo', shape=[16 * 20 * 16, 100], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
		}
	with tf.variable_scope(name_or_scope='enc3') as scope:
		l3_f1 = tf.get_variable(name='kernel1', shape=[3,3,12, 8], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
		l3_f2 = tf.get_variable(name='kernel2', shape=[5,5,12, 8], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

		l3_b1 = tf.get_variable(name='bias1', shape=[8], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
		l3_b2 = tf.get_variable(name='bias2', shape=[8], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

	
	conv1 = tf.add(conv2d_with_filter('enc1_1', x, l1_f1, [1, 1, 1, 1], [1, 1, 1, 1]), l1_b1)
	conv1 = tf.nn.leaky_relu(conv1, alpha=slope)
	conv2 = tf.add(conv2d_with_filter('enc1_2', x, l1_f2, [1, 1, 1, 1], [1, 1, 1, 1]), l1_b2)
	conv2 = tf.nn.leaky_relu(conv2, alpha=slope)
	conv3 = tf.add(conv2d_with_filter('enc1_3', x, l1_f3, [1, 1, 1, 1], [1, 1, 1, 1]), l1_b3)
	conv3 = tf.nn.leaky_relu(conv3, alpha=slope)
	conv4 = tf.add(conv2d_with_filter('enc1_4', x, l1_f4, [1, 1, 1, 1], [1, 1, 1, 1]), l1_b4)
	conv4 = tf.nn.leaky_relu(conv4, alpha=slope)
	midFeat = tf.concat([conv1, conv2, conv3, conv4], axis=3)
	midFeat = maxpool2d(midFeat, [2, 2], [2, 2])

	conv1 = tf.add(conv2d_with_filter('enc2_1', midFeat, l2_f1, [1, 1, 1, 1], [1, 1, 1, 1]), l2_b1)
	conv1 = tf.nn.leaky_relu(conv1, alpha=slope)
	conv2 = tf.add(conv2d_with_filter('enc2_2', midFeat, l2_f2, [1, 1, 1, 1], [1, 1, 1, 1]), l2_b2)
	conv2 = tf.nn.leaky_relu(conv2, alpha=slope)
	conv3 = tf.add(conv2d_with_filter('enc2_3', midFeat, l2_f3, [1, 1, 1, 1], [1, 1, 1, 1]), l2_b3)
	conv3 = tf.nn.leaky_relu(conv3, alpha=slope)
	midFeat = tf.concat([conv1, conv2, conv3], axis=3)
	midFeat = maxpool2d(midFeat, [2, 2], [2, 2])

	conv1 = tf.add(conv2d_with_filter('enc3_1', midFeat, l3_f1, [1, 1, 1, 1], [1, 1, 1, 1]), l3_b1)
	conv1 = tf.nn.leaky_relu(conv1, alpha=slope)
	conv2 = tf.add(conv2d_with_filter('enc3_2', midFeat, l3_f2, [1, 1, 1, 1], [1, 1, 1, 1]), l3_b2)
	conv2 = tf.nn.leaky_relu(conv2, alpha=slope)
	midFeat = tf.concat([conv1, conv2], axis=3)
	midFeat = maxpool2d(midFeat, [2, 2], [2, 2])

	midFeat = tf.reshape(midFeat, shape=[-1, 16 * 20 * 16])
	output = tf.matmul(midFeat, weights['ew'])
	return output
	
def decoding(x, wo, filters, bias):
	midFeat = tf.nn.leaky_relu(tf.matmul(x, tf.transpose(wo)), alpha=slope)
	midFeat = tf.reshape(midFeat, shape=[-1, 16, 20, 16])

	mid1, mid2 = tf.split(midFeat, [8, 8], 3)
	mid1 = tf.subtract(mid1, bias[2][0])
	mid2 = tf.subtract(mid2, bias[2][1])
	mid1 = conv2d_trans_with_filter('dec3_1', mid1, filters[2][0], [batch_size, 32, 40, 12], [1, 2, 2, 1])
	mid2 = conv2d_trans_with_filter('dec3_2', mid2, filters[2][1], [batch_size, 32, 40, 12], [1, 2, 2, 1])
	midFeat = tf.add_n([mid1, mid2])/2
	
	mid1, mid2, mid3 = tf.split(midFeat, [4, 4, 4], 3)
	mid1 = tf.subtract(mid1, bias[1][0])
	mid2 = tf.subtract(mid2, bias[1][1])
	mid3 = tf.subtract(mid3, bias[1][2])
	mid1 = conv2d_trans_with_filter('dec2_1', mid1, filters[1][0], [batch_size, 64, 80, 8], [1, 2, 2, 1])
	mid2 = conv2d_trans_with_filter('dec2_2', mid2, filters[1][1], [batch_size, 64, 80, 8], [1, 2, 2, 1])
	mid3 = conv2d_trans_with_filter('dec2_3', mid3, filters[1][2], [batch_size, 64, 80, 8], [1, 2, 2, 1])
	midFeat = tf.add_n([mid1, mid2, mid3])/3

	mid1, mid2, mid3, mid4 = tf.split(midFeat, [2, 2, 2, 2], 3)
	mid1 = tf.subtract(mid1, bias[0][0])
	mid2 = tf.subtract(mid2, bias[0][1])
	mid3 = tf.subtract(mid3, bias[0][2])
	mid4 = tf.subtract(mid4, bias[0][3])
	mid1 = conv2d_trans_with_filter('dec1_1', mid1, filters[0][0], [batch_size, 128, 160, 1], [1, 2, 2, 1])
	mid2 = conv2d_trans_with_filter('dec1_2', mid2, filters[0][1], [batch_size, 128, 160, 1], [1, 2, 2, 1])
	mid3 = conv2d_trans_with_filter('dec1_3', mid3, filters[0][2], [batch_size, 128, 160, 1], [1, 2, 2, 1])
	mid4 = conv2d_trans_with_filter('dec1_4', mid4, filters[0][3], [batch_size, 128, 160, 1], [1, 2, 2, 1])
	output = tf.add_n([mid1, mid2, mid3, mid4])/4

	return output

# make a graph
g = tf.Graph()
with g.as_default() as graph:
# load
	# define place holder
	X = tf.placeholder(tf.float32, [batch_size, nFrq, nFrm, 1])
   	P = tf.placeholder(tf.float32, [batch_size])
 
	# Encoding
	code = encoding(X, initDir)

	# collect trainable parameters
	wo = graph.get_tensor_by_name('E/wo:0')
	enc1_f1 = graph.get_tensor_by_name('enc1/kernel1:0')
	enc1_f2 = graph.get_tensor_by_name('enc1/kernel2:0')
	enc1_f3 = graph.get_tensor_by_name('enc1/kernel3:0')
	enc1_f4 = graph.get_tensor_by_name('enc1/kernel4:0')
	enc1_b1 = graph.get_tensor_by_name('enc1/bias1:0')
	enc1_b2 = graph.get_tensor_by_name('enc1/bias2:0')
	enc1_b3 = graph.get_tensor_by_name('enc1/bias3:0')
	enc1_b4 = graph.get_tensor_by_name('enc1/bias4:0')
	
	enc2_f1 = graph.get_tensor_by_name('enc2/kernel1:0')
	enc2_f2 = graph.get_tensor_by_name('enc2/kernel2:0')
	enc2_f3 = graph.get_tensor_by_name('enc2/kernel3:0')
	enc2_b1 = graph.get_tensor_by_name('enc2/bias1:0')
	enc2_b2 = graph.get_tensor_by_name('enc2/bias2:0')
	enc2_b3 = graph.get_tensor_by_name('enc2/bias3:0')

	enc3_f1 = graph.get_tensor_by_name('enc3/kernel1:0')
	enc3_f2 = graph.get_tensor_by_name('enc3/kernel2:0')
	enc3_b1 = graph.get_tensor_by_name('enc3/bias1:0')
	enc3_b2 = graph.get_tensor_by_name('enc3/bias2:0')

	filters = [[enc1_f1, enc1_f2, enc1_f3, enc1_f4], [enc2_f1, enc2_f2, enc2_f3], [enc3_f1, enc3_f2]]
	bias    = [[enc1_b1, enc1_b2, enc1_b3, enc1_b4], [enc2_b1, enc2_b2, enc2_b3], [enc3_b1, enc3_b2]]

	# binary sample for code
	#:wncode = tf.math.l2_normalize(code,axis=1)
	prob  = tf.nn.sigmoid(code)
	code_sample = extractCodeSample(prob)

	# Decoding
	X_rec = decoding(code_sample, wo, filters, bias)

	# make a variable list for training
	t_vars = tf.trainable_variables()

	# define error
	rmse = tf.losses.mean_squared_error(labels=X, predictions=X_rec)
	node_cons = tf.square(tf.subtract(tf.reduce_sum(prob,1), P))
	node_cons = tf.reduce_mean(node_cons)
	cost = rmse + lambda_reg*node_cons

	# optimize the cost
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
	train_op = optimizer.minimize(cost, var_list=t_vars)
	model = [wo, enc1_f1, enc1_f2, enc1_f3, enc1_f4, enc1_b1, enc1_b2, enc1_b3, enc1_b4, enc2_f1, enc2_f2, enc2_f3, enc2_b1, enc2_b2, enc2_b3, enc3_f1, enc3_f2, enc3_b1, enc3_b1]
    
with tf.Session(graph=g) as sess:
	sess.run(tf.global_variables_initializer())
    
	nTrnBatch = int(nTrnData/batch_size)
	nValBatch = int(nValData/batch_size)
	cons = avg_act_node*np.ones(batch_size)

	vcost = 100
	for epoch in range(max_epoch):
		data_indices = np.arange(nTrnData)
		np.random.shuffle(data_indices)
		TrnData = TrnData[data_indices]
		trncost = 0
		trnrmse = 0
		for bter in range(nTrnBatch):
			sidx = bter*batch_size
			eidx = (bter+1)*batch_size
			batchData = TrnData[sidx:eidx]
			_, steprmse, stepcost = sess.run([train_op, rmse, cost], feed_dict={X: batchData, P: cons})
			trncost += stepcost
			trnrmse += steprmse
		trncost /= nTrnBatch
		trnrmse /= nTrnBatch

		data_indices = np.arange(nValData)
		np.random.shuffle(data_indices)
		ValData = ValData[data_indices]
		valcost = 0
		valrmse = 0
		for bter in range(nValBatch):
			sidx = bter*batch_size
			eidx = (bter+1)*batch_size
			batchData = ValData[sidx:eidx]
			steprmse, stepcost = sess.run([rmse, cost], feed_dict={X: batchData, P: cons})
			valcost += stepcost
			valrmse += steprmse
		valcost /= nValBatch
		valrmse /= nValBatch

		logging.info("[Epoch %d] train cost rmse: %f %f, validation cost rmse: %f %f ", epoch, trncost, trnrmse, valcost, valrmse)

		if valcost < vcost:
			## saving model
			msg = "[Epoch %d] train cost rmse: %f %f, validation cost rmse: %f %f " % (epoch, trncost, trnrmse, valcost, valrmse)
			saveParams(sess, paramsDir, msg)
			vcost = valcost
