import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import database_processing as dp

tf.logging.set_verbosity(tf.logging.INFO)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return np.float32(np.array(a[p])), np.int32(np.array(b[p]))

def get_train_data():

	# train dataset
	train_elginvector_data = []
	train_gesture_label = []
	for i in range(0,91):
		train_elginvector_data.append( dp.db_extract_one_elginvector_data(i))
		train_gesture_label.append( dp.db_extract_one_gesture_label(i))

	for i in range(101,191):
		train_elginvector_data.append( dp.db_extract_one_elginvector_data(i))
		train_gesture_label.append( dp.db_extract_one_gesture_label(i))


	for i in range(201,291):
		train_elginvector_data.append( dp.db_extract_one_elginvector_data(i))
		train_gesture_label.append( dp.db_extract_one_gesture_label(i))

	train_data = np.float32(np.array(test_elginvector_data))
	train_target = np.int32(np.array(test_gesture_label))

	return unison_shuffled_copies(train_data , train_target)

	# train_elginvector_data = tf.data.Dataset.from_tensor_slices(np.array(train_elginvector_data))
	# train_gesture_label = tf.data.Dataset.from_tensor_slices(np.array(train_gesture_label)).map(lambda z: tf.one_hot(z, 3))
	# train_dataset = tf.data.Dataset.zip((train_elginvector_data, train_gesture_label)).shuffle(500)

	# return train_dataset

def get_test_data():

	test_elginvector_data = []
	test_gesture_label = []
	for i in range(0,91):
		test_elginvector_data.append( dp.db_extract_one_elginvector_data(i))
		test_gesture_label.append( dp.db_extract_one_gesture_label(i))

	for i in range(101,191):
		test_elginvector_data.append( dp.db_extract_one_elginvector_data(i))
		test_gesture_label.append( dp.db_extract_one_gesture_label(i))


	for i in range(201,291):
		test_elginvector_data.append( dp.db_extract_one_elginvector_data(i))
		test_gesture_label.append( dp.db_extract_one_gesture_label(i))

	# test_elginvector_data = tf.data.Dataset.from_tensor_slices(np.array(test_elginvector_data))
	# test_gesture_label = tf.data.Dataset.from_tensor_slices(np.array(test_gesture_label)).map(lambda z: tf.one_hot(z, 3))
	# test_dataset = tf.data.Dataset.zip((test_elginvector_data, test_gesture_label)).shuffle(500)
	test_data = np.array(test_elginvector_data)
	test_target = np.array(test_gesture_label)

	return unison_shuffled_copies(test_data , test_target)

def add_layer(inputs, in_size, out_size, activeation_function=None):
	Weights = tf.Variable(tf.random.normal([in_size,out_size]))
	biases = tf.Variable(tf.zeros[1,out_size]+0.1)


def main(unused_argv):

	# train_dataset = get_train_data()
	train_data ,train_target = get_test_data() 
	test_data ,test_target = get_test_data() 

	print test_target

	validation_metrics = {
		"accuracy":
			tf.contrib.learn.MetricSpec(
			metric_fn=tf.contrib.metrics.streaming_accuracy,
			prediction_key="classes"),
		"precision":
			tf.contrib.learn.MetricSpec(
			metric_fn=tf.contrib.metrics.streaming_precision,
			prediction_key="classes"),
		"recall":
			tf.contrib.learn.MetricSpec(
			metric_fn=tf.contrib.metrics.streaming_recall,
			prediction_key="classes")
	}

	validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
		test_data,
		test_target,
		every_n_steps=50,
		metrics=validation_metrics,
		early_stopping_metric="loss",
		early_stopping_metric_minimize=True,
		early_stopping_rounds=200)	

	# Specify that all features have real-value data
	feature_columns = [tf.contrib.layers.real_valued_column("", dimension=3)]

	# Build 3 layer DNN with 10, 20, 10 units respectively.
	classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
		hidden_units=[5, 5, 5],
		n_classes=3,
		model_dir="/Users/chenweiyu/Desktop/hand_SVM/myo_data/ten",
		config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))
	  # # Fit model.
	classifier.fit(x=train_data,y=train_target,steps=10,monitors=[validation_monitor])

	accuracy_score = classifier.evaluate(x=test_data, y=test_target)["accuracy"]
	print("Accuracy: {0:f}".format(accuracy_score))

	# new_samples = np.array(
	# 	[[6.4, 3.2, 4.5], [5.8, 3.1, 5.0]], dtype=np.float32)
	# y = list(classifier.predict(new_samples))
	# print("Predictions: {}".format(str(y)))

if __name__ == '__main__':

	tf.app.run()



