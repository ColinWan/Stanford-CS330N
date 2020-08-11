import numpy as np
import random
import tensorflow as tf
from load_data import DataGenerator
from tensorflow.python.platform import flags
from tensorflow.keras import layers

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')

flags.DEFINE_integer('num_samples', 1,
                     'number of examples used for inner gradient update (K for K-shot learning).')

flags.DEFINE_integer('meta_batch_size', 16,
                     'Number of N-way classification tasks per batch')


def loss_function(preds, labels):
    """
    Computes MANN loss
    Args:
        preds: [B, K+1, N, N] network output
        labels: [B, K+1, N, N] labels
    Returns:
        scalar loss
    """
    #############################
    #### YOUR CODE GOES HERE ####

    loss = tf.keras.losses.categorical_crossentropy(preds[:,-1:], labels[:,-1:])
    return tf.reduce_mean(loss)
    #############################


class MANN(tf.keras.Model):

    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)
        self.layer3 = tf.keras.layers.Activation('softmax')
        self.layer_reshape = tf.keras.layers.Reshape((samples_per_class, num_classes, num_classes))

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        _, K, N, D = input_images.shape
        input_img_scaled = tf.reshape(input_images, (-1, K*N, D))
        input_label_scaled = tf.reshape(input_labels, (-1, K*N, N))[:,:-1,:]
        input_label_scaled = tf.concat((input_label_scaled, tf.zeros_like(input_label_scaled[:,-1:])), 1)
        input_label_scaled = tf.dtypes.cast(input_label_scaled, tf.float32)


        x = tf.concat((input_img_scaled, input_label_scaled), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.layer_reshape(x)
        #############################
        return out

data_generator = DataGenerator(
    FLAGS.num_classes, FLAGS.num_samples + 1)
o = MANN(FLAGS.num_classes, FLAGS.num_samples + 1)
optim = tf.keras.optimizers.Adam(0.001)

def train_step(images, labels):
    with tf.GradientTape() as tape:
        pred = o(images, labels)
        loss = loss_function(pred, labels)
    gradients = tape.gradient(loss, o.trainable_variables)
    optim.apply_gradients(zip(gradients, o.trainable_variables))
    return loss

for step in range(50000):
    image, label = data_generator.sample_batch('train', FLAGS.meta_batch_size)
    ls = train_step(image, label)

    if step % 100 == 0:
        print("*" * 5 + "Iter " + str(step) + "*" * 5)
        i, l = data_generator.sample_batch('test', 100)

        pred = o(i, l)
        tls = loss_function(pred, l)
        print("Train Loss:", ls, "Test Loss:", tls)
        pred = tf.reshape(pred,
            [-1, FLAGS.num_samples + 1,
            FLAGS.num_classes, FLAGS.num_classes])
        pred = tf.argmax(pred[:, -1, :, :], axis=2)
        l = l[:, -1, :, :].argmax(2)

        print("Test Accuracy", (1.0 * (np.array(pred) == l)).mean())

