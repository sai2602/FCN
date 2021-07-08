from common import inputs
from common import depthmap_helpers
from common import pointcloud_helpers
from common import CsvReader
from sklearn import preprocessing
import numpy as np
import cv2
import glob
from pathlib import Path, PureWindowsPath



def save_depthmap_image(path, prefix):
    non_workpiece_color = (255, 0, 0)
    workpiece_color = (0, 0, 255)

    #count = 0
    for file in glob.glob(path + "/*/*" +prefix):
        # if count < 26:
        #     count+=1
        #     continue
        print("Working on ", file)
        depthmap, labels = CsvReader.CsvReader.load(file, lambda value: value)

        w = labels.shape[0]
        h = labels.shape[1]
        img = np.zeros((w, h, 3), np.uint8)

        z_max = np.max(np.max(depthmap))
        z_min = np.min(np.min(depthmap))
        z_scale = z_max - z_min

        for x in range(w):
            for y in range(h):
                if labels[x, y] != -1:

                    # calc color imgGrouhdTruth
                    if labels[x, y] == 0:
                        color_true = workpiece_color
                    else:
                        color_true = non_workpiece_color
                    img[x, y] = color_true

        # We don't need to see the whole picture
        #deltaHeight = 0
        # cv2.imwrite(input_path+f+'_groundTruth.png', imgTrue[deltaHeight:, :])
        print("saving Depth Map Image" + file[:-4]+".png")
        cv2.imwrite(file[:-4]+".png", img)

def save_depthmap_grayscale(path, prefix):

    # count = 0
    for file in glob.glob(path + "/*/*" + prefix):
        # if count < 26:
        #     count+=1
        #     continue
        print("Working on ", file)
        depthmap, labels = CsvReader.CsvReader.load(file, lambda value: value)

        w = labels.shape[0]
        h = labels.shape[1]
        img = np.zeros((w, h, 1), np.uint8)

        z_max = np.max(np.max(depthmap))
        z_min = np.min(np.min(depthmap))
        z_scale = z_max - z_min

        for x in range(w):
            for y in range(h):
                img[x, y] = depthmap[x,y]

        # We don't need to see the whole picture
        # deltaHeight = 0
        # cv2.imwrite(input_path+f+'_groundTruth.png', imgTrue[deltaHeight:, :])
        print("saving Depth Map Image" + file[:-4] + ".png")
        cv2.imwrite(file[:-4] + "_grayscale.png", img)

def save_depthmap_image_blurred(path, prefix,blur_k,z_flip=False):
    non_workpiece_color= (255, 0, 0) #
    workpiece_color = (0, 0, 255)
    black = (0, 0, 0)
    for file in glob.glob(path + "/*/*" +prefix):
        print("Working on ", file)
        depthmap, labels = CsvReader.CsvReader.load(file, lambda value: value)
        if z_flip:
            masked_value = np.max(np.max(depthmap))
        else:
            masked_value = 0
        #if prefix.endswith("blurred.csv"):
        depthmap, labels = depthmap_helpers.blur_depth_image_with_labels(depthmap, labels,
                                                                                       blur_k, masked_value)
        # w = labels.shape[0]
        # h = labels.shape[1]
        # img = np.zeros((w, h, 3), np.uint8)
        # img_nn = np.zeros((w,h,3),np.uint8)
        # labels_nn = labels
        # z_max = np.max(np.max(depthmap))
        # z_min = np.min(np.min(depthmap))
        # z_scale = z_max - z_min
        #
        # for x in range(w):
        #     for y in range(h):
        #         # calc color imgGrouhdTruth
        #         if labels[x, y] == 0.0:
        #             color_true = workpiece_color
        #             color_nn = workpiece_color
        #             labels_nn[x,y] = 1.0
        #         elif labels[x, y] == 1.0:
        #             color_true = non_workpiece_color
        #             color_nn = non_workpiece_color
        #             labels_nn[x,y] = 0.0
        #         else:
        #             color_nn = non_workpiece_color
        #             color_true = black
        #             labels_nn[x,y] = 0.0
        #
        #         img[x, y] = color_true
        #         img_nn[x,y] = color_nn

        # We don't need to see the whole picture
        #deltaHeight = 0
        # cv2.imwrite(input_path+f+'_groundTruth.png', imgTrue[deltaHeight:, :])
        # print("saving Depth Map Image ", file[:-4]+"_"+str(blur_k)+"_blurred.png")
        # cv2.imwrite(file[:-4]+"_"+str(blur_k)+"_blurred.png", img)
        CsvReader.CsvReader.save(depthmap, labels, file[:-4]+"_"+str(blur_k)+"_blurred.csv")
        print("Saved")
        #cv2.imwrite(file[:-4] + outFilePrefix + "_nnInput.png", img_nn)
        #CsvReader.CsvReader.save(depthmap, labels_nn, file[:-4] + outFilePrefix + "_nnInput.csv")