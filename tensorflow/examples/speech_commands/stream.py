# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Runs a trained audio graph against a WAVE file and reports the results.

The model, labels and .wav file specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audio model from Python.

Here's an example of running it:

python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import tensorflow as tf
import numpy as np
import librosa


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

FLAGS = None


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(sess, labels, wav_data, input_layer, softmax_tensor, wav_data_tensor, wave_file, num_top_predictions):
  """Runs the audio data through the graph and prints predictions."""
  # Feed the audio data as input to the graph.
  #   predictions  will contain a two-dimensional array, where one
  #   dimension represents the input image count, and the other has
  #   predictions per class
  
  predictions, = sess.run(softmax_tensor, feed_dict={
      input_layer: wav_data,
      wav_data_tensor: wave_file
    })

  # Sort to show labels in order of confidence
  top_k = predictions.argsort()[-num_top_predictions:][::-1]
  results = []
  for node_id in top_k:
    human_string = labels[node_id]
    score = predictions[node_id]
    results.append((human_string, score))
  return results

import time
import io
import matplotlib.pyplot as plt
import librosa.display

def stream(wav, labels, graph, input_name, output_name, how_many_labels, data_dir):
  """Loads the model and labels, and runs the inference to print predictions."""
  # if not wav or not tf.gfile.Exists(wav) and not data_dir:
  #   tf.logging.fatal('Audio file does not exist %s', wav)

  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)

  labels_list = load_labels(labels)

  # load graph, which is stored in the default session
  load_graph(graph)

  wave_file = open('/tmp/speech_dataset/_background_noise_/white_noise.wav', 'rb').read()
  wav_data, sr = librosa.load(wav, sr=44100)
  total_samples = len(wav_data)

  sess = tf.InteractiveSession()
  
  wav_data_tensor = sess.graph.get_tensor_by_name(input_name)
  softmax_tensor = sess.graph.get_tensor_by_name(output_name)
  audio_data_tensor = sess.graph.get_tensor_by_name('decoded_sample_data:0')
  
  window_size = 4410
  increment = 4410 #.1 seconds 
  sound = MovingSoundWindow(wav_data)

  plt.figure(figsize=(12, 8))
  start = time.time()

  for i in range(0, total_samples, increment):
    wav_data, window_start, window_end = sound.next_window()
    audio_data_tensor.data = wav_data
    
    if i >= window_size: #wait to accumulate a full second of samples
      name, confidence = run_graph(sess, labels_list, wav_data.reshape(window_size, 1), audio_data_tensor, softmax_tensor, wav_data_tensor, wave_file, 1)[0]
      
      if confidence > .5 or 'silence' in name:
        print (name, confidence, 'at', window_start, '-', window_end)

  print('total elapsed:', time.time() - start)


class MovingSoundWindow(np.ndarray):

  window_width = 4410
  increment = 4410

  def __new__(cls, inputarr):
    obj = np.asarray(inputarr).view(cls)
    return obj

  def __init__(self, *args, **kwargs):
    self.position = 0
    self.current_window = np.array([])
    self.step = 0
    self.rolling = False
    super(MovingSoundWindow, self).__init__()

  def next_window(self, data=None):
    self.step += 1
    window = self[self.position: self.increment * self.step]

    if self.rolling or len(window) >= self.window_width:
      self.rolling = True # save this to avoid length calculation
      self.position += self.increment

    # return the samples, and the start/end of the window
    if self.step and not self.step % 10:
      print (self[self.position - (self.increment * self.step): self.position])
      D = librosa.amplitude_to_db(
        self[self.position - (self.increment * self.step): self.position]
      )
      librosa.display.specshow(D, y_axis='log')
      plt.savefig('graphs/%d.png'%self.position)
    return window, self.position, self.position + (self.increment * self.step)
  

def main(_):
  """Entry point for script, converts flags to arguments."""
  stream(FLAGS.wav, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
            FLAGS.output_name, FLAGS.how_many_labels, FLAGS.data_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--wav', type=str, default='', help='Audio file to be identified.')
  parser.add_argument(
      '--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument(
      '--labels', type=str, default='', help='Path to file containing labels.')
  parser.add_argument(
      '--input_name',
      type=str,
      default='wav_data:0',
      help='Name of WAVE data input node in model.')
  parser.add_argument(
      '--output_name',
      type=str,
      default='labels_softmax:0',
      help='Name of node outputting a prediction in the model.')
  parser.add_argument(
      '--how_many_labels',
      type=int,
      default=3,
      help='Number of results to show.')
  parser.add_argument(
      '--data_dir',
      type=str,
      default='',
      help="""\
      test predictions on this directory.
      """)

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
