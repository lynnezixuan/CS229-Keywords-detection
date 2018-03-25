from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def build_nn(model_input, model_params, set_model,
                 is_training, runtime_settings=None):
  """select a model to build: single_fc, fc_dnn, cnn"""
  if set_model == 'vanilla':
    return create_vanilla_model(model_input, model_params,is_training)
  elif set_model == 'fc_dnn':
    return create_fc_dnn_model(model_input,model_params,is_training)
  elif set_model == 'cnn':
    return create_cnn_model(model_input,model_params,is_training)
  else:
    raise Exception('model not supported, should be one of "vanilla", "fc_dnn", "cnn"')


def weight_variable(shape, dev):
  initial = tf.truncated_normal(shape, stddev = dev)
  return tf.Variable(initial)

def create_vanilla_model(model_input, model_params, is_training):
  """Build a single fully-connected layer model"""
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_size = model_params['input_size']
  label_count = model_params['label_count']
  weights = weight_variable([input_size, label_count], 0.001)
  bias = tf.Variable(tf.zeros([label_count]))
  logits = tf.matmul(model_input, weights) + bias
  if is_training:
    return logits, dropout_prob
  else:
    return logits

def create_fc_dnn_model(model_input, model_params, is_training):
  """Build a standard fully-connected deep neural nets"""
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_size = model_params['input_size']
  label_count = model_params['label_count']
  #define hidden units number per layer
  heights = tf.constant(128,dtype=tf.int32)
  #initialize weights and bias
  weights1 = weight_variable([input_size, heights], 0.1)
  bias1 = tf.Variable(tf.zeros([heights]))
  weights2 = weight_variable([heights, heights],0.1)
  bias2 = tf.Variable(tf.zeros([heights]))
  weights3 = weight_variable([heights, heights],0.1)
  bias3 = tf.Variable(tf.zeros([heights]))
  weights4 = weight_variable([heights, label_count], 0.1)
  bias4 = tf.Variable(tf.zeros([label_count]))
  #hidden layer 1
  hidden1 = tf.matmul(model_input, weights1) + bias1
  hidden1_relu = tf.nn.relu(hidden1)
  if is_training:
    hidden1_dropout = tf.nn.dropout(hidden1_relu,dropout_prob)
  else:
    hidden1_dropout = hidden1_relu
 # hidden layer 2
  hidden2 = tf.matmul(hidden1_dropout,weights2) + bias2
  hidden2_relu = tf.nn.relu(hidden2)
  if is_training:
    hidden2_dropout = tf.nn.dropout(hidden2_relu,dropout_prob)
  else:
    hidden2_dropout = hidden2_relu
  #hidden layer 3
  hidden3 = tf.matmul(hidden2_dropout,weights3) + bias3
  hidden3_relu = tf.nn.relu(hidden3)
  if is_training:
    hidden3_dropout = tf.nn.dropout(hidden3_relu, dropout_prob)
  else:
    hidden3_dropout = hidden3_relu
  #output layer
  logits = tf.matmul(hidden3_dropout,weights4) + bias4

  if is_training:
    return logits, dropout_prob
  else:
    return logits


def create_cnn_model(model_input, model_params, is_training):
  """Builds a ConvNets model"""
  if is_training:
    dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
  input_frequency_size = model_params['dct_coefficient_count']
  input_time_size = model_params['spectrogram_length']
  fingerprint_4d = tf.reshape(model_input, [-1, input_time_size, input_frequency_size, 1])
  #1st convolutional layer
  first_filter_width = 8
  first_filter_height = 20
  first_filter_count = 64
  first_weights = weight_variable([first_filter_height, first_filter_width, 1, first_filter_count], 0.1)
  first_bias = tf.Variable(tf.zeros([first_filter_count]))
  first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [1,1,1,1],
                            'SAME') + first_bias
  first_relu = tf.nn.relu(first_conv)
  if is_training:
    first_dropout = tf.nn.dropout(first_relu, dropout_prob)
  else:
    first_dropout = first_relu
  #max pooling on frequency
  max_pool = tf.nn.max_pool(first_dropout, [1, 1, 3, 1], [1, 1, 3, 1], 'SAME')
  #2nd convolutional layer
  second_filter_width = 4
  second_filter_height = 10
  second_filter_count = 64
  second_weights = weight_variable([second_filter_height, second_filter_width, first_filter_count,second_filter_count],0.1)
  second_bias = tf.Variable(tf.zeros([second_filter_count]))
  second_conv = tf.nn.conv2d(max_pool, second_weights, [1,1,1,1],
                             'SAME') + second_bias
  second_relu = tf.nn.relu(second_conv)
  if is_training:
    second_dropout = tf.nn.dropout(second_relu, dropout_prob)
  else:
    second_dropout = second_relu
  second_conv_shape = second_dropout.get_shape()
  second_conv_output_width = second_conv_shape[2]
  second_conv_output_height = second_conv_shape[1]
  second_conv_element_count = int(
      second_conv_output_width * second_conv_output_height *
      second_filter_count)
  flattened_second_conv = tf.reshape(second_dropout, [-1, second_conv_element_count])
  #linear low-rank
  lin_heights = tf.constant(32,dtype=tf.int32)
  lin_weights = weight_variable([second_conv_element_count, lin_heights], 0.01)
  lin_bias = tf.Variable(tf.zeros([lin_heights]))
  lin_output = tf.matmul(flattened_second_conv, lin_weights) + lin_bias

  #dnn layer
  dnn_heights = tf.constant(128,dtype=tf.int32)
  dnn_weights = weight_variable([lin_heights,dnn_heights],0.01)
  dnn_bias = tf.Variable(tf.zeros([dnn_heights]))
  dnn_output = tf.matmul(lin_output,dnn_weights) + dnn_bias
  dnn_relu = tf.nn.relu(dnn_output)
  if is_training:
    dnn_dropout = tf.nn.dropout(dnn_relu, dropout_prob)
  else:
    dnn_dropout = dnn_relu

  label_count = model_params['label_count']
  #output layer
  final_fc_weights = weight_variable([dnn_heights, label_count], 0.01)
  final_fc_bias = tf.Variable(tf.zeros([label_count]))
  final_fc = tf.matmul(dnn_dropout, final_fc_weights) + final_fc_bias
  if is_training:
    return final_fc, dropout_prob
  else:
    return final_fc
