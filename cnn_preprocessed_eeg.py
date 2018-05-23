from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from eeg_parser import get_eeg_samples
from eeg_parser import train_test_split_shuffle
from eeg_preprocessing import eeg_fft_plot
from eeg_preprocessing  import eeg_power_spectral_density_plot
from eeg_preprocessing import eeg_fir_bandpass_samples
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


def eeg_cnn_model_preprocessed_fn(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	# Reshape X to 3-D tensor: [batch_size, width, height, channels]
	# eeg_signals are 640 pixels, and have three color channel


	input_layer = tf.reshape(features["x"], [-1, 1, 320, 3])
	input_layer = tf.cast(input_layer, tf.float32)
	#TODO: Move input layer reshaping to outside model function
	print("\nINPUT LAYER SHAPE: \n", str(input_layer.shape))


	# Convolutional Layer #1
	# Computes 32 features using a 5x5 filter with ReLU activation.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 640, 3]
	# Output Tensor Shape: [batch_size, 640, 96]
	print (input_layer[1])
	conv1 = tf.layers.separable_conv2d(
		inputs=input_layer, 
		filters=32, 
		kernel_size=5, 
		padding="same", 
		activation=tf.nn.relu
	)
	# conv1 = tf.layers.conv2d(
	#     inputs=input_layer,
	#     filters=32,
	#     kernel_size=[5, 1],
	#     padding="same",
	#     activation=tf.nn.relu)

	print("\nCONV1 OUTPUT SHAPE: \n", str(conv1.shape))

	# Pooling Layer #1
	# First max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 640, 96]
	# Output Tensor Shape: [batch_size, 320, 96]
	pool1 = tf.layers.max_pooling2d(
	  inputs=conv1, 
	  pool_size=[1, 2], 
	  strides=[1, 2])


	print("\nPOOL1 OUTPUT SHAPE: \n", str(pool1.shape))

	# Convolutional Layer #2
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 320, 96]
	# Output Tensor Shape: [batch_size, 192]
	conv2 = tf.layers.separable_conv2d(
		inputs=pool1, 
		filters=64, 
		kernel_size=5, 
		padding="same", 
		activation=tf.nn.relu
	)
	print("\nCONV2 OUTPUT SHAPE: \n", str(conv2.shape))


	# Pooling Layer #2
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 320, 192]
	# Output Tensor Shape: [batch_size, 160, 192]
	pool2 = tf.layers.max_pooling2d(
	  inputs=conv2, 
	  pool_size=[1, 2], 
	  strides=[1, 2])
	print("\nPOOL2 OUTPUT SHAPE: \n", str(pool2.shape))

		# Convolutional Layer #2
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 320, 96]
	# Output Tensor Shape: [batch_size, 192]
	conv3 = tf.layers.separable_conv2d(
		inputs=pool2, 
		filters=128, 
		kernel_size=5, 
		padding="same", 
		activation=tf.nn.relu
	)
	print("\nCONV3 OUTPUT SHAPE: \n", str(conv3.shape))


	# Pooling Layer #2
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 320, 192]
	# Output Tensor Shape: [batch_size, 160, 192]
	pool3 = tf.layers.max_pooling2d(
	  inputs=conv3, 
	  pool_size=[1, 2], 
	  strides=[1, 2])
	print("\nPOOL3 OUTPUT SHAPE: \n", str(pool3.shape))

		# Convolutional Layer #2
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 320, 96]
	# Output Tensor Shape: [batch_size, 192]
	conv4 = tf.layers.separable_conv2d(
		inputs=pool3, 
		filters=256, 
		kernel_size=5, 
		padding="same", 
		activation=tf.nn.relu
	)
	print("\nCONV4 OUTPUT SHAPE: \n", str(conv4.shape))


	# Pooling Layer #2
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 320, 192]
	# Output Tensor Shape: [batch_size, 160, 192]
	pool4 = tf.layers.max_pooling2d(
	  inputs=conv4, 
	  pool_size=[1, 2], 
	  strides=[1, 2])
	print("\nPOOL4 OUTPUT SHAPE: \n", str(pool4.shape))



	# Flatten tensor into a batch of vectors
	# Input Tensor Shape: [batch_size, 160, 192]
	# Output Tensor Shape: [batch_size, 160 * 192]
	pool4_flat = tf.reshape(pool4, [-1, pool4.shape[1]* pool4.shape[2] * pool4.shape[3]])
	print("\nPOOL4_FLAT OUTPUT SHAPE: \n", str(pool4_flat.shape))
	

	# Dense Layer
	# Densely connected layer with 1024 neurons
	# Input Tensor Shape: [batch_size, 160 * 192]
	# TODO: find out the number of neurons
	# Output Tensor Shape: [batch_size, 1024]
	dense = tf.layers.dense(
	  inputs=pool4_flat, 
	  units=1024, 
	  activation=tf.nn.relu)
	print("\nDENSE OUTPUT SHAPE: \n", str(dense.shape))


	# Add dropout operation; 0.6 probability that element will be kept
	dropout = tf.layers.dropout(
	    inputs=dense, 
	    rate=0.4, 
	    training=mode == tf.estimator.ModeKeys.TRAIN)
	print("\nDROPOUT OUTPUT SHAPE: \n", str(dropout.shape))

	# Logits layer
	# Input Tensor Shape: [batch_size, 1024]
	# Output Tensor Shape: [batch_size, 10]
	logits = tf.layers.dense(
	  inputs=dropout, 
	  units=2)
	print("\nLOGITS OUTPUT SHAPE: \n", str(logits.shape))

	predictions = {
	    # Generate predictions (for PREDICT and EVAL mode)
	    "classes": tf.argmax(input=logits, axis=1),
	    # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
	    # `logging_hook`.
	    "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
	  return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sparse_softmax_cross_entropy(
		labels=labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
	  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
	  train_op = optimizer.minimize(
	      loss=loss,
	      global_step=tf.train.get_global_step())
	  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
	    "accuracy": tf.metrics.accuracy(
	        labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(
	    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

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
	# filtered_samples = eeg_fir_bandpass_samples(eeg_samples)

	# test = 153
	# plt.plot(eeg_samples[test].signal[1])
	# plt.plot(filtered_samples[test].signal[1])
	# plt.show()
	# number_samples_used = len(eeg_samples)
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

	train_y = np.asarray(train_y, dtype = np.int32)
	test_y = np.asarray(test_y, dtype = np.int32)

	print("SAMPLE SHAPE: ", train_x[1].shape)
	print("EEG TRAIN DATA SHAPE: ", str(train_x.shape))
	print("EEG EVAL DATA SHAPE: ", str(test_x.shape))
	print("EEG TRAIN LABELS SHAPE: ", str(train_y.shape))
	print("EEG EVAL LABELS SHAPE: ", str(test_y.shape))


	start_time = time.time()

	# Create the Estimator
	eeg_classifier = tf.estimator.Estimator(
		model_fn=eeg_cnn_model_preprocessed_fn, model_dir="/tmp/eeg_convnet_model")

	# Set up logging for predictions
	# Log the values in the "Softmax" tensor with label "probabilities"
	tensors_to_log2 = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log2, every_n_iter=50)


	accuracies = []
	train_accuracies = []
	num_runs = 5
	steps_completed = 0
	steps_per_train = 100
	accuracies.append(0.5)
	train_accuracies.append(0.5)


	# Train the model
	for i in range(num_runs):
		train_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": train_x},
			y=train_y,
			batch_size=20,
			num_epochs=None,
			shuffle=True)
		eeg_classifier.train(
			input_fn=train_input_fn,
			steps=steps_per_train,
			hooks=[logging_hook])

		steps_completed += steps_per_train
		# Evaluate the model and print results
		train_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": train_x},
			y=train_y,
			num_epochs=1,
			shuffle=True)

		eval_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": test_x},
			y=test_y,
			num_epochs=1,
			shuffle=True)


		print("\n\nDONE TRAINING for run #", i + 1, ", NOW EVALUATE\n\n")
		train_results = eeg_classifier.evaluate(input_fn=train_eval_input_fn)
		eval_results = eeg_classifier.evaluate(input_fn=eval_input_fn)
		print("hey, here are the evaluation results: ",eval_results)
		print("accuracy is: ", str(eval_results['accuracy']))
		train_accuracies.append(train_results['accuracy'])
		accuracies.append(eval_results['accuracy'])

	print(accuracies)
	print("completed ", str(steps_completed), "training steps in ", str(int((time.time() - start_time)/60)), "minutes")

	# Note that using plt.subplots below is equivalent to using
	# fig = plt.figure() and then ax = fig.add_subplot(111)
	fig, ax = plt.subplots()
	ax.plot(accuracies, "r")
	ax.plot(train_accuracies, "b")

	ax.set(xlabel='run number', ylabel='accuracy (%)',
	   title='Test Results')
	ax.grid()


	fig.savefig("test.png")
	plt.show()

if __name__ == "__main__":
	tf.app.run()
