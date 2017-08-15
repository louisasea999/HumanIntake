import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math
import pylab as P
import tensorflow as tf

training_csv = pd.read_csv('MNIST_data/foods_training.csv')
testing_csv = pd.read_csv('MNIST_data/foods_testing.csv')

training_label_csv = pd.read_csv('MNIST_data/foods_training_label.csv')
testing_label_csv = pd.read_csv('MNIST_data/foods_testing_label.csv')

training_data = np.array(training_csv, dtype='float32')
testing_data = np.array(testing_csv, dtype='float32')
training_label_data = np.array(training_label_csv, dtype='float32')
testing_label_data = np.array(testing_label_csv, dtype='float32')
print(testing_data[0,:])
print(testing_data[1,:])
print(testing_data[2,:])
print(testing_data[3,:])
print(testing_data[4,:])
print(testing_data[5,:])
print(testing_data[6,:])
pdata = tf.placeholder("float", [None,4])
plabel = tf.placeholder("float", [4])
distance = tf.reduce_sum(tf.abs(tf.add(pdata,tf.negative(plabel))), reduction_indices=1)
pred = tf.arg_min(distance, 0)
accuracy = 0.
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	for i in range(len(testing_data)):
		nn_index = sess.run(pred, feed_dict={pdata: training_data, plabel: testing_data[i, :]})
		print("Test", i,"nn_index is",nn_index, "Prediction:", np.argmax(training_label_data[nn_index]), "True Class:", np.argmax(testing_label_data[i])," prediction class is ",training_label_data[nn_index]," but the true class is ",testing_label_data[i])
		if (training_label_data[nn_index] == testing_label_data[i]):
			accuracy += 1./len(testing_data)
	print("Done!")
	print("Accuracy:", accuracy)