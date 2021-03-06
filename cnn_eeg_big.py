from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from eeg_parser import get_eeg_data

tf.logging.set_verbosity(tf.logging.INFO)


def eeg_cnn_model_big_fn(features, labels, mode):
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
      filters=64,
      kernel_size=[10, 1],
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
      kernel_size=[7, 1],
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
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=96,
      kernel_size=[5, 1],
      padding="same",
      activation=tf.nn.relu)
  print("\nCONV3 OUTPUT SHAPE: \n", str(conv3.shape))


  # Pooling Layer #3
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 320, 192]
  # Output Tensor Shape: [batch_size, 160, 192]
  pool3 = tf.layers.max_pooling2d(
    inputs=conv3, 
    pool_size=[2, 1], 
    strides=[2,1])
  print("\nPOOL3 OUTPUT SHAPE: \n", str(pool3.shape))


  # Convolutional Layer #4
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 320, 96]
  # Output Tensor Shape: [batch_size, 192]
  conv4 = tf.layers.conv2d(
      inputs=pool3,
      filters=128,
      kernel_size=[5, 1],
      padding="same",
      activation=tf.nn.relu)
  print("\nCONV4 OUTPUT SHAPE: \n", str(conv4.shape))

  # Pooling Layer #3
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 320, 192]
  # Output Tensor Shape: [batch_size, 160, 192]
  pool4 = tf.layers.max_pooling2d(
    inputs=conv4, 
    pool_size=[2, 1], 
    strides=[2,1])
  print("\nPOOL4 OUTPUT SHAPE: \n", str(pool4.shape))

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 160, 192]
  # Output Tensor Shape: [batch_size, 160 * 192]
  pool4_flat = tf.reshape(pool4, [-1, pool4.shape[1]* 3 * 128])
  print("\nPOOL3_FLAT OUTPUT SHAPE: \n", str(pool4_flat.shape))
  

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
  print ("\nYOOOO\n")
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  print("\nHELLOOOOO\n")
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    print("howdy\n")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.002)
    print("partner\n")
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    print("fuckoff\n")
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  print("\nWHATTSSUUPPPP\n")
  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  data = get_eeg_data()
  eeg_train_data = data[0]
  eeg_train_labels = data[1] - 1
  eeg_eval_data = data[2]
  eeg_eval_labels = data[3] - 1
  # eeg_data = data[0]
  # eeg_labels = data[1] - 1
  # print("EEG DATA SHAPE: ", str(data[0].shape))
  # print("EEG LABELS SHAPE: ", str(data[1].shape))
  # eeg_train_data = eeg_data[:5000,:]
  # eeg_train_labels = eeg_labels[:5000]
  # eeg_eval_data = eeg_data[5000:, :]
  # eeg_eval_labels = eeg_labels[5000:]
  print("EEG TRAIN DATA SHAPE: ", str(eeg_train_data.shape))
  print("EEG EVAL DATA SHAPE: ", str(eeg_eval_data.shape))
  print("EEG TRAIN LABELS SHAPE: ", str(eeg_train_labels.shape))
  print("EEG EVAL LABELS SHAPE: ", str(eeg_eval_labels.shape))

  # Load training and eval data
  # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  # train_data = mnist.train.images  # Returns np.array
  # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  # eval_data = mnist.test.images  # Returns np.array
  # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)


  # print ('train_data shape: {}'.format(str(train_data.shape)))

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=eeg_cnn_model_big_fn, model_dir="/tmp/mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log2 = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log2, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eeg_train_data},
      y=eeg_train_labels,
      batch_size=20,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=500,
      hooks=[logging_hook])

  print("\n\nDONE TRAINING: NOW EVALUATE\n\n")

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eeg_eval_data},
      y=eeg_eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()