"""
Simple TensorFlow exercises
You should thoroughly test your code
"""

import tensorflow as tf

###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################

x = tf.random_uniform([])  # Empty array as shape creates a scalar.
y = tf.random_uniform([])
out = tf.cond(tf.less(x, y), lambda: tf.add(x, y), lambda: tf.subtract(x, y))
with tf.Session() as sess:
    x,y,out = sess.run([x,y,out])
    print("=========1a==========")
    print("x: ", x)
    print("y: ", y)
    print("out: ", out)
###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from -1 and 1.
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################
x_1b = tf.random_uniform([], minval=-1, maxval=1)
y_1b = tf.random_uniform([], minval=-1, maxval=1)

predicate_dict = {tf.less(x_1b, y_1b) : lambda :tf.add(x_1b, y_1b), tf.greater(x_1b,y_1b): lambda :tf.subtract(x_1b,y_1b)}

out_1b = tf.case(predicate_dict, default= lambda : tf.constant(0, dtype=tf.float32))
with tf.Session() as sess:
    x_1b, y_1b, out_1b = sess.run([x_1b, y_1b, out_1b])
    print("=========1b===========")
    print("x :", x_1b)
    print("y :", y_1b)
    print("out: ", out_1b)

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]]
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################
x_1c = tf.constant([[0, -2, -1], [0, 1, 2]], dtype=tf.float32)
y_1c = tf.zeros_like(x)
out_1c = tf.equal(x_1c, y_1c)
with tf.Session() as sess:
    out_1c = sess.run(out_1c)
print("=========1c===========")
print(out_1c)

###############################################################################
# 1d: Create the tensor x of value
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################

x_1d = tf.constant([29.05088806,  27.61298943,  31.19073486,  29.35532951,
  30.97266006,  26.67541885,  38.08450317,  20.74983215,
  34.94445419,  34.45999146,  29.06485367,  36.01657104,
  27.88236427,  20.56035233,  30.20379066,  29.51215172,
  33.71149445,  28.59134293,  36.05556488,  28.66994858])

out_1d = tf.where(tf.greater(x_1d, 30))

elements = tf.gather(x_1d, out_1d)

with tf.Session() as sess:
    out_1d = sess.run([out_1d, elements])
    print("===========1d==============")
    print(out_1d)
    print(elements)

###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################

x_1e = tf.diag(tf.range(1,7,1))
x_1e = tf.diag(tf.range(1,7,1))

with tf.Session() as sess:
    x_1e = sess.run(x_1e)
    print("=========1e==========")
    print(x_1e)

###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################

x_1f = tf.truncated_normal([10,10])

deter = tf.matrix_determinant(x_1f)

with tf.Session() as sess:
    deter = sess.run(deter)
    print("=========1f==========")
    print(deter)

###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################

x_1g = tf.constant([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9])

unique_array = tf.unique(x_1g)
with tf.Session() as sess:
    unique_array = sess.run(unique_array)
    print("=========1g==========")
    print(unique_array[0])

###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.less() and tf.select() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: see the Huber loss function in the lecture slides 3.
###############################################################################

x_1h = tf.truncated_normal([300])
y_1h = tf.truncated_normal([300])

x_1h_sub_y_1h = tf.subtract(x_1h, y_1h)

mean_x_sub_y = tf.reduce_mean(x_1h_sub_y_1h)
square_x_sub_y = tf.square(mean_x_sub_y)
mse = tf.reduce_mean(square_x_sub_y)

sum_abs = tf.reduce_sum(tf.abs(x_1h_sub_y_1h))

out_1h = tf.cond(tf.less(mean_x_sub_y, 0.0), lambda : mse, lambda : sum_abs)

with tf.Session() as sess:
    mean_x_sub_y, out_1h = sess.run([mean_x_sub_y, out_1h])
    print("============1h=============")
    print(mean_x_sub_y)
    print(out_1h)