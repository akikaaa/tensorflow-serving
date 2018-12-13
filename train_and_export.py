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
    x_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x')
    y_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')
    w = tf.Variable(name='w', initial_value=0.0)
    z = tf.Variable(name='z', initial_value=0.0)
    y = w*x_ph + z
    loss_op = tf.losses.mean_squared_error(y_ph, y)
    train_op = tf.train.GradientDescentOptimizer(3e-4).minimize(loss_op)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(FLAGS.training_iteration):
        data_xs = []
        data_ys = []
        for _ in range(FLAGS.batch_size):
            data_x = random.uniform(-100.0, 100.0)
            data_y = a*data_x + b + 0.2*random.gauss(0, 1.5)
            data_xs.append(data_x)
            data_ys.append(data_y)
        data_xs = np.array(data_xs).reshape([-1, 1])
        data_ys = np.array(data_ys).reshape([-1, 1])
        _,  loss, w_, z_ = sess.run([train_op, loss_op, w, z], feed_dict={x_ph: data_xs, y_ph: data_ys})
        if i % 100 == 0:
            tf.logging.info('step: %s, loss: %s, w: %s, z: %s' % (i, loss, w_, z_))

    # EXPORT
    export_path_base = 'export'
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(FLAGS.model_version)))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    tensor_info_x = tf.saved_model.utils.build_tensor_info(x_ph)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y)
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'x': tensor_info_x},
            outputs={'y': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
    )
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict':
                prediction_signature
        }
    )
    builder.save()


if __name__ == '__main__':
    main()
