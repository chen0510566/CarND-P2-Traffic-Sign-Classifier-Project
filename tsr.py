# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = './traffic-signs-data/train.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train_and_valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train_and_valid['features'], train_and_valid['labels'],
                                                      test_size=0.2)
X_test, y_test = test['features'], test['labels']

# save 5 images
# import random
# import scipy.misc
# for i in range(0, 5):
#     scipy.misc.imsave('image'+str(i)+'.jpg', X_valid[random.randint(0, 7842)])

print('X_train shape: {}, y_train shape: {}'.format(X_train.shape, y_train.shape))
print('X_valid shape: {}, y_valid shape: {}'.format(X_valid.shape, y_valid.shape))
print('X_test shape: {}, y_test shape: {}'.format(X_test.shape, y_test.shape))

import matplotlib.pyplot as plt
# image=X_train[0]
# plt.figure(figsize=(1, 1))
# plt.imshow(image)
# plt.show()

# plt.figure()
# plt.hist(y_test, 43)
# plt.title('historgram of traffic signs')
# plt.xlabel('category index')
# plt.ylabel('num of traffic signs')
# plt.show()

import numpy as np
from sklearn.utils import shuffle


def rgb2gray(rgb):
    r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return np.expand_dims(gray, axis=3) / 255


X_train = rgb2gray(X_train)
X_valid = rgb2gray(X_valid)
X_test = rgb2gray(X_test)

X_train, y_train = shuffle(X_train, y_train)

import tensorflow as tf

EPOCH = 15
BATCH_SIZE = 128

from tensorflow.contrib.layers import flatten


def LeNet(x, keep_prob):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x12
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 12), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(12))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x12. Output = 14x14x12.
    conv1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Input = 14x14x12, Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 12, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # add dropout to prevent overfitting
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    return logits


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1))
y = tf.placeholder(tf.int32, shape=(None))
one_hot_y = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32)

rate = 0.001

logits = LeNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

with tf.Session() as sess:
    # summary_writer = tf.summary.FileWriter('./tmp/mnist_logs', sess.graph)
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    print('Training...\n')

    for i in range(EPOCH):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})

        training_accuracy = evaluate(X_train, y_train)
        validation_accuracy = evaluate(X_valid, y_valid)
        print('EPOCH {} ...'.format(i + 1))
        print('Training Accuracy = {:.3f}\n'.format(training_accuracy))
        print('Validation Accuracy = {:.3f}\n'.format(validation_accuracy))

    saver.save(sess, './lenet')
    print('Model saved')

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

import numpy as np

filelist = 'image0.jpg', 'image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg'
images = np.array([np.array(plt.imread(name)) for name in filelist])
images = rgb2gray(images)

# plt.figure(figsize=(1, 1))
# plt.imshow((images[0]*255).squeeze(), cmap='gray')
# plt.show()


saver = tf.train.Saver()
image_labels = [14, 5, 14, 1, 5]
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(images, image_labels)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

top_k = tf.nn.top_k(tf.nn.softmax(logits), k=5)
image_labels = [14, 5, 14, 1, 5]
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(images, image_labels)
    top_k_prob = sess.run(top_k, feed_dict={x: images, keep_prob: 1.0})
    print("Images Test Accuracy = {:.3f}".format(test_accuracy))
    print('Top 3 softmax value: ')
    print(top_k_prob.values)
    print('Corresponding labels:')
    print(top_k_prob.indices)
