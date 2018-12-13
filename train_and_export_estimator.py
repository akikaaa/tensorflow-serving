import tensorflow as tf
import random
import numpy as np
import os
tf.flags.DEFINE_integer('training_iteration', 18000, 'number of training iterations.')
tf.flags.DEFINE_integer('batch_size', 120, 'batch size')
tf.flags.DEFINE_string('work_dir', 'tmp', 'Working directory.')
tf.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.flags.FLAGS

def main():
    # TRAIN
    # Y = aX+b
    # true regression coefficient
    a = 2.0
    b = 4.5
    tf.logging.set_verbosity(tf.logging.INFO)
    features = []
    labels = []
    for _ in range(5000):
        data_x = random.uniform(-100.0, 100.0)
        data_y = a * data_x + b + 0.2 * random.gauss(0, 1.5)
        features.append(data_x)
        labels.append(data_y)
    features = {'x': features}

    def train_input_fn(features, labels, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        dataset = dataset.shuffle(1000).repeat().batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()

    def model_fn(features, labels, mode):
        x = features['x']
        w = tf.Variable(name='w', initial_value=0.0)
        z = tf.Variable(name='z', initial_value=0.0)
        y = w * x + z
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'predictions': y,
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        loss = tf.losses.mean_squared_error(labels, y)
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss)
        assert mode == tf.estimator.ModeKeys.TRAIN
        optimizer = tf.train.GradientDescentOptimizer(1e-4)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.work_dir
    )

    estimator.train(input_fn=lambda: train_input_fn(features, labels, FLAGS.batch_size))

    # EXPORT
    feature_spec = {
        "x": tf.FixedLenFeature([1], tf.float32),
    }

    # Build receiver function, and export.
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
    estimator.export_savedmodel('export', serving_input_receiver_fn)


if __name__ == '__main__':
    main()
