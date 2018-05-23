from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from eeg_parser import get_eeg_samples
from eeg_parser import train_test_split_shuffle
from eeg_preprocessing import eeg_fft_plot
from eeg_preprocessing  import eeg_power_spectral_density_plot
from eeg_preprocessing import eeg_fir_bandpass_plot
from eeg_preprocessing import eeg_fir_bandpass
from eeg_preprocessing import process_data
from eeg_preprocessing  import energy_percents
from eeg_preprocessing import input_energy_graph
from eeg_preprocessing import energy_band_percent_graphs
import matplotlib.pyplot as plt
import time
import pywt
import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)

def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x, W, [1, 1, 1, 1], padding='VALID')

def apply_depthwise_conv(x,kernel_size,num_channels,depth):
    weights = weight_variable([1, kernel_size, num_channels, depth])
    print("WEIGHTS SHAPE: ", weights.shape)
    biases = bias_variable([depth * num_channels])
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights),biases))

def apply_max_pool(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1], 
                          strides=[1, 1, stride_size, 1], padding='VALID')

def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)

def stack_sigs(data): 

	segments = np.empty((0,len(data[0].signal[0]),3))
	labels = np.empty((0))
	i = 0
	for sample in data:
		i += 1
		if (i % 100 == 0): 
			print(i)

		sig = sample.signal
		ch1 = sig[0]#.tolist()
		ch2 = sig[1]#.tolist()
		ch3 = sig[2]#.tolist()

		segments = np.vstack([segments, np.dstack([ch1, ch2, ch3])])
		labels = np.append(labels, sample.label)

	return segments, labels

def main(unused_argv):
	# Load training and eval data"
	print("starting")
	eeg_samples = get_eeg_samples('SeniorProject/EEG_Dataset/')
	number_samples_used = 1000
	print("TOTAL NUMBER OF SAMPLES: ", number_samples_used)
	train_samples, test_samples = train_test_split_shuffle(eeg_samples, number_samples_used, 0.7)
	print("stacking training sigs...")
	train_x, train_y = stack_sigs(train_samples)
	print("stacking testing sigs..." )
	test_x, test_y = stack_sigs(test_samples)

	print("TRAIN_X: ", train_x.shape)
	print("TRAIN_Y: ", train_y.shape)
	train_x = train_x.reshape(len(train_x), 1, 320, 3)
	test_x = test_x.reshape(len(test_x), 1, 320, 3)

	train_y = np.asarray(pd.get_dummies(train_y), dtype = np.int8)
	test_y = np.asarray(pd.get_dummies(test_y), dtype = np.int8)

	print("SAMPLE SHAPE: ", train_x[1].shape)
	print("EEG TRAIN DATA SHAPE: ", str(train_x.shape))
	print("EEG EVAL DATA SHAPE: ", str(test_x.shape))
	print("EEG TRAIN LABELS SHAPE: ", str(train_y.shape))
	print("EEG EVAL LABELS SHAPE: ", str(test_y.shape))

	input_height = 1
	input_width = 320
	num_labels = 2
	num_channels = 3

	batch_size = 10
	kernel_size = 10
	depth = 60
	num_hidden = 1000

	learning_rate = 0.007
	training_epochs = 8

	train_accuracies = [0.5]
	test_accuracies = [0.5]

	total_batches = train_x.shape[0] // batch_size
	# total_batches = 10
	print("total batches: ", total_batches)


	X = tf.placeholder(tf.float32, shape=[None,input_height,input_width,num_channels])
	Y = tf.placeholder(tf.float32, shape=[None,num_labels])

	c = apply_depthwise_conv(X,kernel_size,num_channels,depth)
	print("\nCONV1 OUTPUT SHAPE: \n", str(c.shape))

	p = apply_max_pool(c,20,2)
	print("\nPOOL1 OUTPUT SHAPE: \n", str(p.shape))

	c = apply_depthwise_conv(p,6,depth*num_channels,depth//10)
	print("\nCONV2 OUTPUT SHAPE: \n", str(c.shape))

	shape = c.get_shape().as_list()
	c_flat = tf.reshape(c, [-1, shape[1] * shape[2] * shape[3]])
	print("\nCONV2_FLAT OUTPUT SHAPE: \n", str(c_flat.shape))

	f_weights_l1 = weight_variable([shape[1] * shape[2] * depth * num_channels * (depth//10), num_hidden])
	f_biases_l1 = bias_variable([num_hidden])
	f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1),f_biases_l1))

	out_weights = weight_variable([num_hidden, num_labels])
	out_biases = bias_variable([num_labels])
	y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)
	print("y_ SHAPE: ", y_.shape)

	loss = -tf.reduce_sum(Y * tf.log(y_))
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

	correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	cost_history = np.empty(shape=[1],dtype=float)

	with tf.Session() as session:
		tf.global_variables_initializer().run()
		for epoch in range(training_epochs):
			for b in range(total_batches):    
				print("batch: ", b)
				offset = (b * batch_size) % (train_y.shape[0] - batch_size)
				batch_x = train_x[offset:(offset + batch_size), :, :, :]
				batch_y = train_y[offset:(offset + batch_size), :]
				_, c = session.run([optimizer, loss],feed_dict={X: batch_x, Y : batch_y})
				cost_history = np.append(cost_history,c)

			print("calculating training accuracies...")
			train_accuracy = session.run(accuracy, feed_dict={X: train_x, Y: train_y})
			train_accuracies.append(train_accuracy)
			print ("Epoch: ",epoch," Training Loss: ",c," Training Accuracy: ", train_accuracy)
			test_accuracy = session.run(accuracy, feed_dict={X: test_x, Y: test_y})
			test_accuracies.append(test_accuracy)

			fig, ax = plt.subplots()
			ax.plot(test_accuracies, "r")
			ax.plot(train_accuracies, "b")

			ax.set(xlabel='run number', ylabel='accuracy (%)',
			   title='Test Results')
			ax.grid()
			plt.show()



		print("calculating testing accuracies")
		print ("Testing Accuracy:", session.run(accuracy, feed_dict={X: test_x, Y: test_y}))

	fig, ax = plt.subplots()
	ax.plot(test_accuracies, "r")
	ax.plot(train_accuracies, "b")

	ax.set(xlabel='run number', ylabel='accuracy (%)',
	   title='Test Results')
	ax.grid()


	fig.savefig("test.png")
	plt.show()

if __name__ == "__main__":
	tf.app.run()