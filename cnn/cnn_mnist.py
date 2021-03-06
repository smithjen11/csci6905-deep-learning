from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import skimage
from skimage import data
from skimage import transform
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.color import rgb2gray
import random


tf.logging.set_verbosity(tf.logging.INFO)

ROOT_PATH = "some_path"


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 3])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      data_format="channels_last",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      data_format="channels_last",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=256)

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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
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


def load_training_data(data_directory):
  directories = [d for d in os.listdir(data_directory) 
                 if os.path.isdir(os.path.join(data_directory, d))]
  labels = []
  images = []
  label_names = []
  count = 0
  for d in directories:
    label_names.append(d)
    label_directory = os.path.join(data_directory, d)
    file_names = [os.path.join(label_directory, f) 
                  for f in os.listdir(label_directory) 
                  if f.endswith(".jpg")]

    # Train on a subset of data
    for f in file_names[:20]:
      images.append(skimage.data.imread(f))
      labels.append(count)

    count = count + 1
  return images, labels


def load_eval_data(data_directory):
  directories = [d for d in os.listdir(data_directory) 
                 if os.path.isdir(os.path.join(data_directory, d))]
  labels = []
  images = []
  label_names = []
  count = 0
  for d in directories:
    label_names.append(d)
    label_directory = os.path.join(data_directory, d)
    file_names = [os.path.join(label_directory, f) 
                  for f in os.listdir(label_directory) 
                  if f.endswith(".jpg")]

    for f in file_names:
      images.append(skimage.data.imread(f))
      labels.append(count)

    count = count + 1
  return images, labels


def main(unused_argv):
  # Load training and eval data
  data_directory = os.path.join(ROOT_PATH, "256_ObjectCategories")
  
  ## Training data
  images, labels = load_training_data(data_directory)

  # Resize images
  images28 = [transform.resize(image, (28, 28, 3)) for image in images]
  images28 = np.array(images28, dtype=np.float32)

  train_data = images28  # Returns np.array
  train_labels = np.asarray(labels, dtype=np.int32)

  ## Eval data
  eval_images, eval_labels = load_eval_data(data_directory)

  # Resize images
  eval_images28 = [transform.resize(image, (28, 28, 3)) for image in eval_images]
  eval_images28 = np.array(eval_images28, dtype=np.float32)
  eval_data = eval_images28  # Returns np.array
  eval_labels = np.asarray(eval_labels, dtype=np.int32)

  # Create the Estimator
  classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=os.path.join(ROOT_PATH, "cnn_model6"))

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=150)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  classifier.train(
      input_fn=train_input_fn,
      steps=10000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
