from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from eeg_parser import get_eeg_data
import matplotlib.pyplot as plt
import time
from cnn_eeg_big import eeg_cnn_model_big_fn

tf.logging.set_verbosity(tf.logging.INFO)


def eeg_cnn_model_4deep_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 3-D tensor: [batch_size, width, height, channels]
  # eeg_signals are 640 pixels, and have three color channel
  input_layer = tf.reshape(features["x"], [-1, 320, 3])
  input_layer = tf.cast(input_layer, tf.float32)
  print("\nINPUT LAYER SHAPE: \n", str(input_layer.shape))


  # (batch, 128, 9) --> (batch, 64, 18)
  conv1 = tf.layers.conv1d(
    inputs=input_layer, 
    filters=32, 
    kernel_size=2, 
    strides=1,
    padding='same', 
    activation = tf.nn.relu)
  print("\nCONV1 OUTPUT SHAPE: \n", str(conv1.shape))
  max_pool_1 = tf.layers.max_pooling1d(
    inputs=conv1,
    pool_size=2, 
    strides=2, 
    padding='same')
  print("\nPOOL1 OUTPUT SHAPE: \n", str(max_pool_1.shape))
  
  # (batch, 64, 18) --> (batch, 32, 36)
  conv2 = tf.layers.conv1d(
    inputs=max_pool_1, 
    filters=64, 
    kernel_size=2, 
    strides=1, 
    padding='same',
    activation = tf.nn.relu)
  print("\nCONV2 OUTPUT SHAPE: \n", str(conv2.shape))
  max_pool_2 = tf.layers.max_pooling1d(
    inputs=conv2, 
    pool_size=2, 
    strides=2, 
    padding='same')
  print("\nPOOL2 OUTPUT SHAPE: \n", str(max_pool_2.shape))

  
  # (batch, 32, 36) --> (batch, 16, 72)
  conv3 = tf.layers.conv1d(
    inputs=max_pool_2, 
    filters=128, 
    kernel_size=2, 
    strides=1,
    padding='same', 
    activation = tf.nn.relu)
  print("\nCONV3 OUTPUT SHAPE: \n", str(conv3.shape))
  max_pool_3 = tf.layers.max_pooling1d(
    inputs=conv3, 
    pool_size=2, 
    strides=2, 
    padding='same')
  print("\nPOOL3 OUTPUT SHAPE: \n", str(max_pool_3.shape))
  
  # (batch, 16, 72) --> (batch, 8, 144)
  conv4 = tf.layers.conv1d(
    inputs=max_pool_3, 
    filters=256, 
    kernel_size=2, 
    strides=1, 
    padding='same', 
    activation = tf.nn.relu)
  print("\nCONV4 OUTPUT SHAPE: \n", str(conv4.shape))
  max_pool_4 = tf.layers.max_pooling1d(
    inputs=conv4, 
    pool_size=2, 
    strides=2, 
    padding='same')
  print("\nPOOL4 OUTPUT SHAPE: \n", str(max_pool_4.shape))

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 160, 192]
  # Output Tensor Shape: [batch_size, 160 * 192]
  pool4_flat = tf.reshape(max_pool_4, [-1, max_pool_4.shape[1]*  256])
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
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.008)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    train_metric_op = {"accuracy": tf.metrics.accuracy(
      labels=labels, 
      predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {"accuracy": tf.metrics.accuracy(
    labels=labels, 
    predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

