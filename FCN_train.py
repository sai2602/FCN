from __future__ import print_function
import os
import tensorflow as tf
import numpy as np

import common.TensorflowUtils as utils
from model import vgg_net, inference, FLAGS, train
import datetime
from common.DatasetReader import read_train_test_list, BatchDatasetReader, normalize_meanstd
from six.moves import xrange
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#MAX_ITERATION = int(1e5 + 1)
MAX_ITERATION = 3000000000
MAX_EPOCHS = 50
IMAGE_SIZE = 1000
TRAIN_FILE = 'train_list.txt'
TEST_FILE = 'test_list.txt'


def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    loss_summary = tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Reading training and test list...")
    train_records = read_train_test_list(FLAGS.data_dir,TRAIN_FILE)
    valid_records = read_train_test_list(FLAGS.data_dir, TEST_FILE)
    print('train records', len(train_records))
    print('valid records', len(valid_records))

    train_dataset_reader = BatchDatasetReader(train_records, image_size=(IMAGE_SIZE, IMAGE_SIZE))
    validation_dataset_reader = BatchDatasetReader(valid_records, image_size=(IMAGE_SIZE, IMAGE_SIZE))
    logs_file = FLAGS.logs_dir + '/logs.txt'
    f_log = open(logs_file, "w+")
    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()

    # create two summary writers to show training loss and validation loss in the same graph
    # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + 'train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + 'validation')

    sess.run(tf.global_variables_initializer())
    #ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    #if ckpt and ckpt.model_checkpoint_path:
        #saver.restore(sess, ckpt.model_checkpoint_path)
        #print("Model restored...")
    #saver.restore(sess, FLAGS.trained_model)
    val_loss = 50000
    if FLAGS.mode == "train":
        print('Training started ...')
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations, epoch_finished = train_dataset_reader.next_batch(FLAGS.batch_size,
                                                                                              annotation_name = 'Contours',
                                                                                              annotation_suffix = '.png',
                                                                                              use_CSV=True)
            train_images = normalize_meanstd(train_images,axis=(1,2))
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                train_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                valid_images, valid_annotations, epoch = validation_dataset_reader.next_batch(FLAGS.batch_size,
                                                                                              annotation_name='Contours',
                                                                                              annotation_suffix='.png',
                                                                                              use_CSV=True)
                valid_images = normalize_meanstd(valid_images, axis=(1, 2))
                valid_loss, summary_sva = sess.run([loss, loss_summary], feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                if valid_loss < val_loss:
                    val_loss = valid_loss
                    # add validation loss to TensorBoard
                    validation_writer.add_summary(summary_sva, itr)
                    saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
                    text = "%s ---> Validation_loss: %g , model : %g " % (datetime.datetime.now(), valid_loss, itr)
                    f_log.write(text)
                    f_log.write('\n')

        f_log.close()






    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)


if __name__ == "__main__":
    tf.app.run()
