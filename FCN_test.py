from __future__ import print_function
import os
import time
from common import depthmap_helpers
import cv2 as cv
from common.CsvReader import CsvReader
from common.DatasetReader import read_train_test_list, BatchDatasetReader, normalize_meanstd
from model import *


trained_model = "logs/model.ckpt-5500"
IMAGE_SIZE = 1000
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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

    valid_records = read_train_test_list(FLAGS.data_dir, TEST_FILE)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    saver.restore(sess, trained_model)
    print("Model restored...")


    for i in range(len(valid_records)):
        file = valid_records[i]
        print('Predicting file', file)
        depthmap, labels1 = CsvReader.load(file)
        sfile = file.rsplit('/', 2)
        file_labels = sfile[0] + '/' + 'Contours' + '/' + sfile[2][:-4] + '.png'
        labels_org = cv.imread(file_labels, cv.IMREAD_GRAYSCALE)
        labels1 = np.where(labels_org == 255, 1, labels_org)
        labels = np.where(depthmap == 0, 0, labels1)

        # depthmap_resize, padding, cropping, iScropping = depthmap_helpers.resize_image(depthmap, min_dim=None,
        #                                                                                max_dim=IMAGE_SIZE,
        #                                                                                mode='square')
        valid_images = np.expand_dims(np.expand_dims(depthmap,axis=0),axis=3)
        valid_annotations = np.expand_dims(np.expand_dims(labels, axis=0), axis=3)

        valid_images = normalize_meanstd(valid_images, axis=(1, 2))

        ts_start = time.time()
        pred = sess.run(pred_annotation,
                        feed_dict={image: valid_images, annotation: valid_annotations, keep_probability: 1.0})
        print('Number of class 1 pixels', np.count_nonzero(pred == 1))
        t_now = time.time()
        print("Prediction done in ", t_now - ts_start)
        pred_org = np.squeeze(pred, axis=(0, 3))
        pred = np.where(pred_org == 1, 255, pred_org)
        file_save = file_labels = sfile[0] + '/' + 'Tests' + '/' + sfile[2][:-4] + '.csv'
        #cv.imwrite(file_save,labels)
        CsvReader.save(depthmap,pred,file_save)


if __name__ == "__main__":
    tf.app.run()

