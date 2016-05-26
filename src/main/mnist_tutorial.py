import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load the MNIST dataset - one_hot=True means the data contains a target-vector
mnist = input_data.read_data_sets("../resources/mnist", one_hot=True)

# MNIST classification model
# build using softmax regression (http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/)
# with tensorflow (https://www.tensorflow.org/versions/r0.8/tutorials/mnist/beginners/index.html)

# placeholder for any mnist image of 784 dimensions in our model
x = tf.placeholder(tf.float32, [None, 784])

# variable containing weights - 10 since we want K=10 classes for the softmax regression
W = tf.Variable(tf.zeros([784, 10]))

# variable containing the bias
b = tf.Variable(tf.zeros([10]))

# y is our model using softmax regression
y = tf.nn.softmax(
    # multiply W with x
    tf.matmul(x, W) + b
)

# y_ is a placeholder for the target values
y_ = tf.placeholder(tf.float32, [None, 10])

# cross entropy between y and y_ - tf.reduce_mean computes the mean over all the examples in the "batch"
cross_entropy = tf.reduce_mean(
    # tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] parameter
    -tf.reduce_sum(
        # tf.log of each element in y and multiply with y_
        y_ * tf.log(y),
        reduction_indices=[1]
    )
)

# train the model to minimize cross entropy using gradient descent with a learning rate of 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# the model is complete, we can initialize the variables
init = tf.initialize_all_variables()

# start a session with initial values
sess = tf.Session()
sess.run(init)

# Each step of the loop, we get a "batch" of one hundred random data points from our training set.
# We run train_step feeding in the batches of data to replace the placeholders.

# Using small batches of random data is called stochastic training -- in this case, stochastic gradient descent.
# Ideally, we'd like to use all our data for every step of training
# because that would give us a better sense of what we should be doing, but that's expensive.
# So, instead, we use a different subset every time. Doing this is cheap and has much of the same benefit.

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# How good were our model predictions?
# tf.argmax is an extremely useful function which gives you the index of the highest entry in a tensor along some axis.
# tf.argmax(y,1) is the label our model thinks is most likely for each input,
# tf.argmax(y_,1) is the correct label.
# we can use tf.equal to check if our prediction matches the truth.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# cast predictions to bits instead of booleans and calculate the mean
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print the accuracy
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))