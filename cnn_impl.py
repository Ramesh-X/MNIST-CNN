import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
import sys


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_file', './data/train.csv', 'Input training file')
flags.DEFINE_string('test_file', './data/test.csv', 'Testing file')
flags.DEFINE_string('save_file', './data/test_result.csv', 'Test Results file')
flags.DEFINE_string('backup', './models/mnist_model.ckpt', 'Directory for storing data')

dtype = tf.float32


def weight_variable(shape, name):
    # return tf.get_variable(name, shape=shape, dtype=dtype, initializer=tf.contrib.layers.xavier_initializer())
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial, name=name, dtype=tf.float32)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, dtype=dtype)
    return tf.Variable(initial, name=name, dtype=dtype)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, k, s):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1],
                          strides=[1, s, s, 1], padding='SAME')


def avg_pool(x, k, s):
    return tf.nn.avg_pool(x, ksize=[1, k, k, 1],
                          strides=[1, s, s, 1], padding='SAME')


def conv_layer(x, r, c, i, o, name):
    W_conv = weight_variable([r, c, i, o], name + '_w')
    b_conv = bias_variable([o], name + '_b')
    return tf.nn.relu(conv2d(x, W_conv) + b_conv)


def inception_layer(x, i, o11, o21, o22, o31, o32, o41, name):    # input, input_layers, output_layers 1 and 2
    conv1x1_1 = conv_layer(x, 1, 1, i, o11, name + 'conv1x1_1')
    conv1x1_2 = conv_layer(x, 1, 1, i, o21, name + 'conv1x1_2')
    conv3x3 = conv_layer(conv1x1_2, 3, 3, o21, o22, name + 'conv3x3')
    conv1x1_3 = conv_layer(x, 1, 1, i, o31, name + 'conv1x1_3')
    conv5x5 = conv_layer(conv1x1_3, 5, 5, o31, o32, name + 'conv5x5')
    max3x3 = max_pool(x, 3, 1)
    conv1x1_4 = conv_layer(max3x3, 1, 1, i, o41, name + 'conv1x1_4')
    return tf.nn.relu(tf.concat([conv1x1_1, conv3x3, conv5x5, conv1x1_4], 3))


def fully_connected(X, layers, out, name):
    pre_l = 0
    i = 1
    for l in layers:
        if not pre_l:
            pre_l = l
            continue
        w_h = weight_variable([pre_l, l], name + '_' + str(i) + '_w')
        b_h = bias_variable([l], name + '_' + str(i) + '_b')
        X = tf.nn.relu(tf.matmul(X, w_h) + b_h)
        pre_l = l
        i += 1
    w_h = weight_variable([pre_l, out], name + '_' + str(i) + '_w')
    b_h = bias_variable([out], name + '_' + str(i) + '_b')
    return tf.nn.softmax(tf.matmul(X, w_h) + b_h)


def read_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = pd.read_csv(FLAGS.train_file)
    target_tr = train['label']
    train = train.drop("label", axis=1).values
    splt = int(len(train)*0.2)
    test = train[:splt]
    train = train[splt:]
    check = pd.read_csv(FLAGS.test_file)
    target_tr = (np.arange(10) == target_tr.values[:, None]).astype(np.float32)
    target_ts = target_tr[:splt]
    target_tr = target_tr[splt:]
    return train, target_tr, test, target_ts, check.values


def main():
    print("Reading Data..")
    train_x, train_y, test_x, test_y, check_x = read_data()
    print('loading data finished!')
    print('training set: ', train_x.shape, ', test set: ', test_x.shape, ', check set: ', check_x.shape, '\n\n')
    print("Creating Model...")
    X = tf.placeholder(dtype, [None, 784])
    Y = tf.placeholder(dtype, [None, 10])
    tr = tf.placeholder(dtype)
    pr = tf.placeholder(dtype)

    x_ = tf.reshape(X, [-1, 28, 28, 1])     # 28 x 28 x 1
    conv = conv_layer(x_, 5, 5, 1, 64, 'conv1')      # 28 x 28 x 64
    conv = conv_layer(conv, 3, 3, 64, 128, 'conv2')   # 28 x 28 x 128
    pool = max_pool(conv, 3, 2)             # 14 x 14 x 128
    norm = tf.contrib.layers.batch_norm(pool)
    conv = inception_layer(norm, 128, 8, 64, 96, 8, 16, 8, 'inc1')  # 14 x 14 x 128
    conv = inception_layer(conv, 128, 64, 96, 128, 16, 32, 32, 'inc2')  # 14 x 14 x 256
    conv = inception_layer(conv, 256, 160, 112, 224, 24, 64, 64, 'inc3')  # 14 x 14 x 512
    pool = max_pool(conv, 3, 2)             # 7 x 7 x 256
    norm = tf.contrib.layers.batch_norm(pool)
    conv = inception_layer(norm, 512, 128, 128, 256, 32, 64, 64, 'inc4')  # 7 x 7 x 512
    pool = avg_pool(conv, 7, 7)             # 1 x 1 x 512

    '''x_ = tf.reshape(X, [-1, 28, 28, 1])
    h_conv1 = conv_layer(x_, 3, 3, 1, 8, 'conv1')
    h_conv2 = conv_layer(h_conv1, 3, 3, 8, 32, 'conv2')
    h_pool1 = max_pool(h_conv2, 2, 2)
    h_norm1 = tf.contrib.layers.batch_norm(h_pool1)
    h_conv3 = conv_layer(h_norm1, 3, 3, 32, 64, 'conv3')
    h_pool2 = max_pool(h_conv3, 2, 2)
    h_conv4 = conv_layer(h_pool2, 3, 3, 64, 128, 'conv4')
    h_pool3 = avg_pool(h_conv4, 7, 7)'''

    cnn = tf.reshape(pool, [-1, 512])
    # cnn = tf.reshape(x_, [-1, 784])
    cnn = tf.nn.dropout(cnn, pr)
    out_y = fully_connected(cnn, [512], 10, 'fc1')

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_y, labels=Y))
    train_step = tf.train.AdamOptimizer(tr).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(out_y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype))

    print("Initializing variables..")
    BATCH_SIZE = 100
    tr_tmp = 1.00e-4
    acc = 0.1
    pre_acc = 0
    epoch = 0
    saver = tf.train.Saver()
    data_size = len(train_x)

    def save_file(name: str):
        ImageId = ['ImageId']
        Label = ['Label']
        for start in range(0, len(check_x), BATCH_SIZE):
            end = start + BATCH_SIZE
            check_y = sess.run(out_y, feed_dict={X: check_x[start:end], pr: 1.0})
            for i in range(len(check_y)):
                ImageId.append(str(len(ImageId)))
                Label.append(str(np.argmax(check_y[i])))
        df = pd.Series(Label, index=ImageId)
        df.to_csv(FLAGS.save_file + name)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        use_previous = 0
        # use the previous model or don't and initialize variables
        if use_previous:
            saver.restore(sess, './models/mnist_model.ckpt-5.data-00000-of-00001')
            print("Model restored.")

        print("Training...")
        sys.stdout.flush()
        while acc < 0.99999:
            epoch += 1
            # Shuffle the data before each training iteration.
            p = np.random.permutation(range(data_size))
            trX, trY = train_x[p], train_y[p]

            # Train in batches of 128 inputs.
            splt = int(data_size * 0.8)
            for start in range(0, splt, BATCH_SIZE):
                end = start + BATCH_SIZE
                sess.run(train_step, feed_dict={X: trX[start:end], Y: trY[start:end], tr: tr_tmp, pr: 0.45})

            # And print the current accuracy on the training data.
            pre_acc = acc
            if acc >= pre_acc:
                # tr_tmp *= 0.95
                saver.save(sess, FLAGS.backup, global_step=epoch)
            else:
                pass
                # tr_tmp *= 1.06
            # tr_tmp = tr_tmp / (1 + 0.009*epoch) if acc > 0.95 else 1.0e-4 if acc < 0.8 else tr_tmp / (1 + 0.02*epoch)
            acc = sess.run(accuracy, feed_dict={X: trX[splt:], Y: trY[splt:], pr: 1.0})
            print('epoch: ', epoch, ', Accuracy: ', acc, ', learning_rate: ', tr_tmp)
            if epoch % 5 == 0:
                acc = sess.run(accuracy, feed_dict={X: test_x, Y: test_y, pr: 1.0})
                print('epoch: ', epoch, ', Test Accuracy: ', acc, ' : Testing on test set finished >>>>>>>>>>>')
                if acc > 0.99:
                    save_file(str(epoch) + '_a' + str(acc))
            sys.stdout.flush()

        print('\n\n\nTraining finished..!')
        save_file('final')


if __name__ == "__main__":
    main()
