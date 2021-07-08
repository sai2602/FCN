import numpy as np
from common.CsvReader import CsvReader
from common.pointcloud_helpers import project_to_depth_image_orthogonal_by_resultion
from common.depthmap_helpers import blur_depth_image_with_labels, resize_image, apply_contour_filter
from scipy.ndimage.interpolation import rotate
import cv2 as cv
import math

class DataAugmentation:
    def __init__(self, depthmap, labels, save_path, img_size = 1000, simulation_data=False):
        self.depthmap = depthmap
        self.labels = labels
        self.save_path = save_path
        self.img_size = img_size
        self.simulation_data = simulation_data

    def copy_original(self,file_name):
        depthmap, labels = self.get_resized_data(self.depthmap, self.labels)
        self.save(depthmap, labels, file_name)

    def flip_lr(self,file_name):
        depthmap = np.fliplr(self.depthmap)
        labels = np.fliplr(self.labels)
        depthmap, labels = self.get_resized_data(depthmap, labels)
        self.save(depthmap,labels,file_name)


    def flip_up(self,file_name):
        depthmap = np.flipud(self.depthmap)
        labels = np.flipud(self.labels)
        depthmap, labels = self.get_resized_data(depthmap, labels)
        self.save(depthmap, labels, file_name)

    def rotate90_1(self,file_name):
        depthmap = np.rot90(self.depthmap, 1)
        labels = np.rot90(self.labels, 1)
        depthmap, labels = self.get_resized_data(depthmap, labels)
        self.save(depthmap, labels, file_name)

    def rotate90_3(self,file_name):
        depthmap = np.rot90(self.depthmap, 3)
        labels = np.rot90(self.labels, 3)
        depthmap, labels = self.get_resized_data(depthmap, labels)
        self.save(depthmap, labels, file_name)

    def transpose(self,file_name):
        depthmap = np.transpose(self.depthmap)
        labels = np.transpose(self.labels)
        depthmap, labels = self.get_resized_data(depthmap, labels)
        self.save(depthmap, labels, file_name)

    def translate(self, file_name, shift=10, direction='right', roll=True):
        assert direction in ['right', 'left', 'down', 'up'], 'Directions should be top|up|left|right'
        depthmap = self.depthmap.copy()
        labels = self.labels.copy()
        if direction == 'right':
            right_slice_depthmap = depthmap[:, -shift:].copy()
            right_slice_labels = labels[:, -shift:].copy()
            depthmap[:, shift:] = depthmap[:, :-shift]
            labels[:, shift:] = labels[:, :-shift]
            if roll:
                depthmap[:, :shift] = np.fliplr(right_slice_depthmap)
                labels[:, :shift] = np.fliplr(right_slice_labels)
            depthmap, labels = self.get_resized_data(depthmap, labels)
            self.save(depthmap, labels, file_name)
        if direction == 'left':
            left_slice_depthmap = depthmap[:, :shift].copy()
            left_slice_labels = labels[:, :shift].copy()
            depthmap[:, :-shift] = depthmap[:, shift:]
            labels[:, :-shift] = labels[:, shift:]
            if roll:
                depthmap[:, -shift:] = left_slice_depthmap
                labels[:, -shift:] = left_slice_labels
            depthmap, labels = self.get_resized_data(depthmap, labels)
            self.save(depthmap, labels, file_name)
        if direction == 'down':
            down_slice_depthmap = depthmap[-shift:, :].copy()
            down_slice_labels = labels[-shift:, :].copy()
            depthmap[shift:, :] = depthmap[:-shift, :]
            labels[shift:, :] = labels[:-shift, :]
            if roll:
                depthmap[:shift, :] = down_slice_depthmap
                labels[:shift, :] = down_slice_labels
            depthmap, labels = self.get_resized_data(depthmap, labels)
            self.save(depthmap, labels, file_name)
        if direction == 'up':
            upper_slice_depthmap = depthmap[:shift, :].copy()
            upper_slice_labels = labels[:shift, :].copy()
            depthmap[:-shift, :] = depthmap[shift:, :]
            labels[:-shift, :] = labels[shift:, :]
            if roll:
                depthmap[-shift:, :] = upper_slice_depthmap
                labels[-shift:, :] = upper_slice_labels
            depthmap, labels = self.get_resized_data(depthmap, labels)
            self.save(depthmap, labels, file_name)

    def rotate(self,file_name, angle = 45, rotation = 'z'):
        depthmap = self.depthmap
        labels = self.labels
        points = np.zeros((depthmap.shape[0]*depthmap.shape[1],4))
        mask_value = 0
        count = 0
        for x in range(depthmap.shape[0]):
            for y in range(depthmap.shape[1]):
                z = depthmap[x,y]
                l = labels[x,y]

                if rotation == 'x':
                    x_rot = x
                    y_rot = y * math.cos(math.radians(angle)) - z * math.sin(math.radians(angle))
                    z_rot = y * math.sin(math.radians(angle)) + z * math.cos(math.radians(angle))
                elif rotation == 'y':
                    x_rot = x * math.cos(math.radians(angle)) + z * math.sin(math.radians(angle))
                    y_rot = y
                    z_rot = z * math.cos(math.radians(angle)) - x * math.sin(math.radians(angle))
                else:
                    x_rot = x * math.cos(math.radians(angle)) - y * math.sin(math.radians(angle))
                    y_rot = x * math.sin(math.radians(angle)) + y * math.cos(math.radians(angle))
                    z_rot = z
                points[count] = [x_rot, y_rot, z_rot, l]
                count+=1
        result_depthmap, result_labels = project_to_depth_image_orthogonal_by_resultion(points,resolution=1,z_flip=False,label_map=0)
        depthmap, labels = blur_depth_image_with_labels(result_depthmap,result_labels,kernel_size=3,masked_value=mask_value)
        depthmap, labels = self.get_resized_data(depthmap,labels)
        self.save(depthmap, labels, file_name)

    def get_resized_data(self,depthmap,labels):
        depthmap_1, padding, cropping, iScropping = resize_image(depthmap, min_dim=None,
                                                                 max_dim=self.img_size,
                                                                 mode='square')
        depthmap_resized = np.zeros((self.img_size, self.img_size))
        labels_resized = np.zeros((self.img_size, self.img_size))
        depthmap_resized[padding[0][0]:self.img_size - padding[0][1],
        padding[1][0]:self.img_size - padding[1][1]] = depthmap
        labels_resized[padding[0][0]:self.img_size - padding[0][1],
        padding[1][0]:self.img_size - padding[1][1]] = labels

        return depthmap_resized, labels_resized

    def save(self,depthmap,labels,file_name):
        print('Saving ', file_name)
        CsvReader.save(depthmap, labels, self.save_path + '/DepthMapsCSV/' + file_name + '.csv')
        contours = apply_contour_filter(labels, simulataion_data=self.simulation_data)
        cv.imwrite(self.save_path + '/Contours/' + file_name + '.png', contours)
        cv.imwrite(self.save_path + '/DepthMapsPNG_8bit/' + file_name + '.png', depthmap.astype(np.uint8))
        cv.imwrite(self.save_path + '/DepthMapsPNG_16bit/' + file_name + '.png', depthmap.astype(np.uint16))






