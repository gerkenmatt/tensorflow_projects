from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from eeg_parser import get_eeg_data
import matplotlib.pyplot as plt
import time
from cnn_preprocessed_eeg import eeg_cnn_model_preprocessed_fn
import pywt

tf.logging.set_verbosity(tf.logging.INFO)


def eeg_cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 3-D tensor: [batch_size, width, height, channels]
  # eeg_signals are 640 pixels, and have three color channel
  input_layer = tf.reshape(features["x"], [-1, 320, 3, 1])
  input_layer = tf.cast(input_layer, tf.float32)
  print("\nINPUT LAYER SHAPE: \n", str(input_layer.shape))

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 640, 3]
  # Output Tensor Shape: [batch_size, 640, 96]
  print (input_layer[1])
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 1],
      padding="same",
      activation=tf.nn.relu)

  print("\nCONV1 OUTPUT SHAPE: \n", str(conv1.shape))

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 640, 96]
  # Output Tensor Shape: [batch_size, 320, 96]
  pool1 = tf.layers.max_pooling2d(
    inputs=conv1, 
    pool_size=[2, 1], 
    strides=[2, 1])


  print("\nPOOL1 OUTPUT SHAPE: \n", str(pool1.shape))

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 320, 96]
  # Output Tensor Shape: [batch_size, 192]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 1],
      padding="same",
      activation=tf.nn.relu)
  print("\nCONV2 OUTPUT SHAPE: \n", str(conv2.shape))


  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 320, 192]
  # Output Tensor Shape: [batch_size, 160, 192]
  pool2 = tf.layers.max_pooling2d(
    inputs=conv2, 
    pool_size=[2, 1], 
    strides=[2,1])
  print("\nPOOL2 OUTPUT SHAPE: \n", str(pool2.shape))


  # Convolutional Layer #3
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 320, 96]
  # Output Tensor Shape: [batch_size, 192]
  # conv3 = tf.layers.conv2d(
  #     inputs=pool2,
  #     filters=64,
  #     kernel_size=[5, 1],
  #     padding="same",
  #     activation=tf.nn.relu)
  # print("\nCONV3 OUTPUT SHAPE: \n", str(conv3.shape))


  # # Pooling Layer #3
  # # Second max pooling layer with a 2x2 filter and stride of 2
  # # Input Tensor Shape: [batch_size, 320, 192]
  # # Output Tensor Shape: [batch_size, 160, 192]
  # pool3 = tf.layers.max_pooling2d(
  #   inputs=conv3, 
  #   pool_size=[2, 1], 
  #   strides=[2,1])
  # print("\nPOOL3 OUTPUT SHAPE: \n", str(pool3.shape))

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 160, 192]
  # Output Tensor Shape: [batch_size, 160 * 192]
  pool2_flat = tf.reshape(pool2, [-1, pool2.shape[1]* 3 * 64])
  print("\nPOOL3_FLAT OUTPUT SHAPE: \n", str(pool2_flat.shape))
  

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 160 * 192]
  # TODO: find out the number of neurons
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(
    inputs=pool2_flat, 
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
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.004)
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

def energy_percents(sig):
	w = pywt.Wavelet('db4')
	cA6, cD6, cD5, cD4, cD3, cD2, cD1 = pywt.wavedec(sig, 'db2', mode='constant', level=6)
	e1 = sum(abs(cD1)**2)/len(cD1)
	e2 = sum(abs(cD2)**2)/len(cD2)
	e3 = sum(abs(cD3)**2)/len(cD3)
	e4 = sum(abs(cD4)**2)/len(cD4)
	e5 = sum(abs(cD5)**2)/len(cD5)
	e6 = sum(abs(cD6)**2)/len(cD6)
	et = e1 + e2 + e3 + e4 + e5 + e6
	return [round(e1/et, 3), round(e2/et, 3), round(e3/et, 3), round(e4/et,3), round(e5/et,3), round(e6/et, 3)]

def process_data(data): 
	processed_data = []

	#preprocessing for signals
	for sig in data:
		processed_signal = []

		#get the energy percentages for each channel of the signal
		ep1 = energy_percents(sig[0:320])
		ep2 = energy_percents(sig[320:640])
		ep3 = energy_percents(sig[640:960])
		processed_signal.append(ep1[3])
		processed_signal.append(ep1[4])
		processed_signal.append(ep1[5])
		processed_signal.append(ep2[3])
		processed_signal.append(ep2[4])
		processed_signal.append(ep2[5])
		processed_signal.append(ep3[3])
		processed_signal.append(ep3[4])
		processed_signal.append(ep3[5])
		processed_data.append(processed_signal)


	# print("Shape of processed data: ", len(processed_data), " x ", len(processed_data[0]))
	return processed_data

def energy_bar_graph(data): 
	e_band1 = []
	e_band2 = []
	e_band3 = []
	e_band4 = []
	e_band5 = []
	e_band6 = []
	e_band7 = []
	e_band8 = []
	e_band9 = []

	print("DATA SHAPE: ", data.shape)


	for i in range(int(len(data) /5)): 
		# print(i)
		e_band1.append(data[i][0])
		e_band2.append(data[i][1])
		e_band3.append(data[i][2])
		e_band4.append(data[i][3])
		e_band5.append(data[i][4])
		e_band6.append(data[i][5])
		e_band7.append(data[i][6])
		e_band8.append(data[i][7])
		e_band9.append(data[i][8])

	# e_band1.sort()
	# e_band2.sort()
	# e_band3.sort()
	# e_band4.sort()
	# e_band5.sort()
	# e_band6.sort()
	# e_band7.sort()
	# e_band8.sort()
	# e_band9.sort()
	print("printing plot")
	e_bands = e_band9 + e_band8 + e_band7 + e_band6 + e_band5 + e_band4 + e_band3 + e_band2 + e_band1 
	plt.bar(range(len(e_bands)), e_bands, align='center', alpha=0.5)
	plt.show()

def main(unused_argv):
	# Load training and eval data"
	print("starting")
	data = get_eeg_data()
	eeg_train_data = data[0]
	eeg_train_labels = data[1] - 1
	eeg_eval_data = data[2]
	eeg_eval_labels = data[3] - 1
	print("EEG TRAIN DATA SHAPE: ", str(eeg_train_data.shape))
	print("EEG EVAL DATA SHAPE: ", str(eeg_eval_data.shape))
	print("EEG TRAIN LABELS SHAPE: ", str(eeg_train_labels.shape))
	print("EEG EVAL LABELS SHAPE: ", str(eeg_eval_labels.shape))

	processed_train_data = np.asarray(process_data(eeg_train_data))
	processed_eval_data = np.asarray(process_data(eeg_eval_data))

	print("processed train data shape: ", str(processed_train_data.shape))
	print("processed eval data shape: ", str(processed_eval_data.shape))


	energy_bar_graph(processed_train_data)

	# w = pywt.Wavelet('db4')
	# print(w)
	# # cA, cD = pywt.dwt(eeg_train_data[2], 'db3')
	# cA6, cD6, cD5, cD4, cD3, cD2, cD1 = pywt.wavedec(eeg_train_data[11], 'db2', mode='constant', level=6)
	# # bA6, bD6, bD5, bD4, bD3, bD2, bD1 = pywt.wavedec(eeg_train_data[11], 'db1', mode='constant', level=6)

	# # fig1 = plt.figure(1) 
	# # plt.plot(cD2, "r")
	# # fig2 = plt.figure(2)
	# # plt.plot(cD5, "b")

	# print(energy_percents(eeg_train_data[0]))
	# print(energy_percents(eeg_train_data[1]))
	# # plt.plot(cD, "b")
	# # fig3 = plt.figure(3)
	# # plt.plot(eeg_train_data[11], "g")
	# plt.show()

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
	num_runs = 20
	steps_completed = 0
	steps_per_train = 250
	accuracies.append(0.5)

	# Train the model
	for i in range(num_runs):
		train_input_fn = tf.estimator.inputs.numpy_input_fn(
		    x={"x": processed_train_data},
		    y=eeg_train_labels,
		    batch_size=100,
		    num_epochs=None,
		    shuffle=True)
		eeg_classifier.train(
		    input_fn=train_input_fn,
		    steps=steps_per_train,
		    hooks=[logging_hook])

		steps_completed += steps_per_train
		# Evaluate the model and print results
		eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		    x={"x": processed_eval_data},
		    y=eeg_eval_labels,
		    num_epochs=1,
		    shuffle=False)

		print("\n\nDONE TRAINING for run #", i + 1, ", NOW EVALUATE\n\n")
		eval_results = eeg_classifier.evaluate(input_fn=eval_input_fn)
		print("hey, here are the evaluation results: ",eval_results)
		print("accuracy is: ", str(eval_results['accuracy']))
		accuracies.append(eval_results['accuracy'])

	print(accuracies)
	print("completed ", str(steps_completed), "training steps in ", str(int((time.time() - start_time)/60)), "minutes")

	# Note that using plt.subplots below is equivalent to using
	# fig = plt.figure() and then ax = fig.add_subplot(111)
	fig, ax = plt.subplots()
	ax.plot(accuracies, "r")

	ax.set(xlabel='run number', ylabel='accuracy (%)',
	     title='Test Results')
	ax.grid()

  # fig.savefig("test.png")
  #plt.show()

if __name__ == "__main__":
  tf.app.run()