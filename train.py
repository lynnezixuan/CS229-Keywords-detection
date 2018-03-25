from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
import argparse
import sys

from six.moves import xrange
import tensorflow as tf
import numpy as np

import input_data
import models

CONFIG = None


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  sess = tf.InteractiveSession()

  #prepare parameters for the model
  audio_length = 1          #audio length = 1 second
  window_size_ms = 30       #analysis window
  window_stride_ms = 10     #window move stride
  sample_rate = 16000        #number of samples per second
  dct_coefficient_count = 40  #length of the coefficient vector

  desired_samples = int(sample_rate * audio_length)
  window_size = int(sample_rate * window_size_ms / 1000)
  window_stride = int(sample_rate * window_stride_ms / 1000)
  spectrogram_length = int(np.maximum(0, desired_samples-window_size) / window_stride) + 1
  input_size = dct_coefficient_count * spectrogram_length
  model_params = {
    'desired_samples': desired_samples,
    'window_size': window_size,
    'window_stride': window_stride,
    'spectrogram_length': spectrogram_length,
    'dct_coefficient_count': dct_coefficient_count,
    'input_size': input_size,
    'label_count': len(input_data.prepare_words_list(CONFIG.wanted_words.split(','))),
    'sample_rate': sample_rate,
  }

  #prepare input data for the model
  silence_percentage = 10.0
  unknown_percentage = 10.0
  validation_percentage = 10
  testing_percentage = 10
  learning_rates_list = [0.001,0.0001]
  momentum_list = [0.5,0.9,0.95,0.99]
  time_shift_ms = 100.0
  audio_processor = input_data.AudioProcessor(
      CONFIG.data_dir, silence_percentage, unknown_percentage,
      CONFIG.wanted_words.split(','), validation_percentage, testing_percentage, model_params)
  input_size = model_params['input_size']
  label_count = model_params['label_count']
  time_shift = int(sample_rate * time_shift_ms / 1000)

  model_input = tf.placeholder(tf.float32, [None, input_size], name='model_input')

  logits, dropout_prob = models.build_nn(
      model_input,
      model_params,
      CONFIG.set_model,
      is_training=True)

  ground_truth = tf.placeholder(
      tf.float32, [None, label_count], name='ground_truth')

  # Back propagation and training evaluation
  with tf.name_scope('cross_entropy'):
    cross_entropy_mean = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
        labels=ground_truth, logits=logits))
  tf.summary.scalar('cross_entropy', cross_entropy_mean)
  with tf.name_scope('train'):
    learning_rate_input = tf.placeholder(tf.float32, [], name='learning_rate_input')
    momentum_input = tf.placeholder(tf.float32, [], name='momentum_input')
    train_step = tf.train.MomentumOptimizer(learning_rate_input,momentum_input,False,"Momentum",True).minimize(cross_entropy_mean)

  expected_results = tf.argmax(ground_truth, 1)
  predicted_results = tf.argmax(logits, 1)
  correct_prediction = tf.equal(predicted_results, expected_results)
  gen_conf_matrix = tf.confusion_matrix(expected_results, predicted_results, num_classes=label_count)
  evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.summary.scalar('accuracy', evaluation_step)

  thresholds = list(np.arange(0,1,0.01))
  true_positive = tf.metrics.true_positives_at_thresholds(ground_truth, tf.nn.softmax(logits), thresholds)
  true_negative = tf.metrics.true_negatives_at_thresholds(ground_truth, tf.nn.softmax(logits), thresholds)
  false_positive = tf.metrics.false_positives_at_thresholds(ground_truth, tf.nn.softmax(logits), thresholds)
  false_negative = tf.metrics.false_negatives_at_thresholds(ground_truth, tf.nn.softmax(logits), thresholds)


  init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  sess.run(init)


  #Training iterations
  total_training_steps = 33000
  evaluate_interval = 400
  BatchSize = 100
  for training_step in xrange(1, total_training_steps + 1):
    #Find the corresponding learning rate and momentum value
    #The first 5/6 training steps use larger learning rate, later 1/6 use a smaller one
    for i in range(len(learning_rates_list)):
      if (training_step <= total_training_steps*5/6):
        learning_rate_value = learning_rates_list[0]
      else:
        learning_rate_value = learning_rates_list[1]
    #Increase momentum value after 1/4 training steps
    for i in range(len(momentum_list)):
      if (training_step > total_training_steps*i/4) and (training_step <= total_training_steps*(i+1)/4):
        momentum_value = momentum_list[i]
        break
    train_input, train_ground_truth = audio_processor.get_data(
        BatchSize, 0, model_params, 0.8,
        0.1, time_shift, 'training', sess)
    train_accuracy, cross_entropy_value, _ = sess.run(
        [ evaluation_step, cross_entropy_mean, train_step ],
        feed_dict={
            model_input: train_input,
            ground_truth: train_ground_truth,
            learning_rate_input: learning_rate_value,
            momentum_input: momentum_value,
            dropout_prob: 0.5
        })

    tf.logging.info('Step #%d: rate %f, momentum %f accuracy %.1f%%, cross entropy %f' %
                    (training_step, learning_rate_value, momentum_value, train_accuracy * 100,
                     cross_entropy_value))
    #Validation
    is_last_step = (training_step == total_training_steps)
    if (training_step % evaluate_interval) == 0 or is_last_step:
      set_size = audio_processor.set_size('validation')
      total_accuracy = 0
      total_conf_matrix = None
      for i in xrange(0, set_size, BatchSize):
        validation_input, validation_ground_truth = (
            audio_processor.get_data(BatchSize, i, model_params, 0.0,
                                     0.0, 0, 'validation', sess))
        validation_accuracy, conf_matrix = sess.run(
            [evaluation_step, gen_conf_matrix],
            feed_dict={
                model_input: validation_input,
                ground_truth: validation_ground_truth,
                dropout_prob: 1.0
            })
        batch_size = min(BatchSize, set_size - i)
        total_accuracy += (validation_accuracy * batch_size) / set_size
        if total_conf_matrix is None:
          total_conf_matrix = conf_matrix
        else:
          total_conf_matrix += conf_matrix
      tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
      tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                      (training_step, total_accuracy * 100, set_size))
      with open("accuracy_trend.txt", 'a', encoding='utf-8') as f:
        f.write('%d %.1f\n' % (training_step, total_accuracy * 100))

  #Testing
  set_size = audio_processor.set_size('testing')
  tf.logging.info('set_size=%d', set_size)
  total_accuracy = 0
  total_cross_entropy_value = 0
  total_true_positive = None
  total_false_positive = None
  total_true_negative = None
  total_false_negative = None
  total_conf_matrix = None
  for i in xrange(0, set_size, BatchSize):
    test_input, test_ground_truth = audio_processor.get_data(
        BatchSize, i, model_params, 0.0, 0.0, 0, 'testing', sess)
    test_cross_entropy_value, test_accuracy, conf_matrix, true_positive_value, true_negative_value, false_positive_value, false_negative_value = sess.run(
        [cross_entropy_mean, evaluation_step, gen_conf_matrix, true_positive, true_negative, false_positive, false_negative],
        feed_dict={
            model_input: test_input,
            ground_truth: test_ground_truth,
            dropout_prob: 1.0
        })
    batch_size = min(BatchSize, set_size - i)
    total_accuracy += (test_accuracy * batch_size) / set_size

    if total_cross_entropy_value == 0:
      total_cross_entropy_value = test_cross_entropy_value
    else:
      total_cross_entropy_value += test_cross_entropy_value
    if total_true_positive is None:
      total_true_positive = true_positive_value
    else:
      total_true_positive = tuple(map(operator.add,total_true_positive,true_positive_value))
    if total_true_negative is None:
      total_true_negative = true_negative_value
    else:
      total_true_negative = tuple(map(operator.add, total_true_negative, true_negative_value))
    if total_false_positive is None:
      total_false_positive = false_positive_value
    else:
      total_false_positive = tuple(map(operator.add, total_false_positive, false_positive_value))
    if total_false_negative is None:
      total_false_negative = false_negative_value
    else:
      total_false_negative = tuple(map(operator.add, total_false_negative, false_negative_value))

    if total_conf_matrix is None:
      total_conf_matrix = conf_matrix
    else:
      total_conf_matrix += conf_matrix
  batch_num = set_size / BatchSize
  final_cross_entropy = total_cross_entropy_value / batch_num
  true_positive_mean = tuple(tp_i / batch_num for tp_i in total_true_positive)
  true_negative_mean = tuple(tn_i / batch_num for tn_i in total_true_negative)
  false_positive_mean = tuple(fp_i / batch_num for fp_i in total_false_positive)
  false_negative_mean = tuple(fn_i / batch_num for fn_i in total_false_negative)
  tp_fn = tuple(map(operator.add,true_positive_mean, false_negative_mean))
  fp_tn = tuple(map(operator.add,false_positive_mean, true_negative_mean))
  tp_fp = tuple(map(operator.add,true_positive_mean, false_positive_mean))
  true_positive_rate = tuple(true_positive_mean[j] / tp_fn[j] for j in range(len(tp_fn)))
  false_positive_rate = tuple(false_positive_mean[k] / fp_tn[k] for k in range(len(fp_tn)))
  precision = tuple(true_positive_mean[h] / tp_fp[h] for h in range(len(tp_fp)))
  tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
  tf.logging.info('Final test accuracy = %.1f%% (N=%d)' % (total_accuracy * 100,set_size))
  tf.logging.info('Final test loss = %.3f' % final_cross_entropy)
  for idx in true_positive_rate:
    with open("ROC.txt", 'a', encoding='utf-8') as fl:
      fl.write(str(idx))
      fl.write("\n")
  for idx in false_positive_rate:
    with open("ROC.txt", 'a', encoding='utf-8') as fl:
      fl.write(str(idx))
      fl.write("\n")
  for idx in precision:
    with open("ROC.txt", 'a', encoding='utf-8') as fl:
      fl.write(str(idx))
      fl.write("\n")


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/speech_dataset/',
      help="""\
      Where to download the speech training data to.
      """)
  parser.add_argument(
      '--wanted_words',
      type=str,
      default='up,down,left,right',
      help='Words to use (others will be added to an unknown label)',)
  parser.add_argument(
      '--set_model',
      type=str,
      default='cnn',
      help='What model architecture to use')

  CONFIG, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
