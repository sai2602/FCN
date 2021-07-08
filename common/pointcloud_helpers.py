import numpy as np
import math

# Maybe a point cloud library (f.e. pcl) should be used in favor of this stuff...
def load(path, values_per_row, n_rows_to_skip, delimiter):
    file = open(path, 'r')
    # Load data in python list
    py_data_list = []
    for line in file:

        # skip first n rows
        if n_rows_to_skip > 0:
            n_rows_to_skip -= 1
            continue

        line = line[:-1]
        tokens = line.split(delimiter)
        if len(tokens) == values_per_row:
            py_data_list += [tokens]

    # Convert python list to np.array
    result = np.zeros([len(py_data_list), values_per_row])
    for i in range(len(py_data_list)):
        for j in range(values_per_row):
            result[i, j] = np.float32(py_data_list[i][j])

    return result


def load_xyz(path, delimiter=';'):
    """
            Loads a point cloud from an ".xyz" file
            :param path: corresponding xyz file
            :param delimiter: delimiter between x, y and z values

            :returns [n,3] np array
    """

    return load(path, 3, 0, delimiter)


def load_pcd(path, delimiter=' '):
    return load(path, 4, 10, delimiter)


def load_csv_with_labels(path, delimiter=','):
    """
            Loads a point cloud from an ".xyz" file
            :param path: corresponding xyz file
            :param delimiter: delimiter between x, y and z values

            :returns [n,4] np array
    """
    return load(path, 4, 0, delimiter)


def write_xyz(path, point_cloud, delimiter=';'):
    """
            Loads a point cloud from an ".xyz" file
            :param path: corresponding xyz file
            :param delimiter: delimiter between x, y and z values

            :returns [n,3] np array
    """
    with open(path, 'w') as file:
        for p in point_cloud:
            file.write('{1}{0}{2}{0}{3}\n'.format(delimiter, p[0], p[1], p[2]))


def getRotationMatrix(rx, ry, rz):
    Tx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ty = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [np.sin(ry), 0, np.cos(ry)]])

    Tz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    return np.dot(np.dot(Tx, Ty), Tz)


def rotate_points(point_cloud, rx, ry, rz):

    print(point_cloud.shape)
    R = getRotationMatrix(rx, ry, rz)
    print(R.shape)

    return np.dot(point_cloud, R)


def project_to_depth_image_perspective(point_cloud, image_width, camera_transformation, focal_length = 1.0):
    """
            Projects a point cloud to a depth image with perspective projection according to the camera-transformation
            and the focal length. Pixels with no point are set to the maximum depth.

            :param point_cloud: (n_points x 3) np array or (n_points x 4) if there is a label per point
            :param image_width: desired with of output image. The height will automatically be calculated according
                to the aspect ratio of the input
            :param camera_transformation: (4 x 4) np array. In bp3 it is saved in scan.cfg in each cylce
                (2 matrices if 2 cameras)
            :param focal_length
            :returns [w_image,h_image] np array
    """

    K = np.zeros(shape=(3, 4))
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[2, 2] = 1.0

    A = np.eye(3)
    labels = point_cloud[:, 3]

    KT = np.matmul(K, camera_transformation)
    AKT = np.matmul(A, KT)
    point_cloud = np.concatenate((point_cloud[:,0:3], np.ones((point_cloud.shape[0], 1))), axis=1)
    point_cloud = np.matmul(point_cloud, AKT.transpose())
    depth = point_cloud[:, 2]

    point_cloud = np.divide(point_cloud[:, 0:2], np.repeat(point_cloud[:, 2:3], 2, axis=1))

    mean = (np.max(point_cloud, axis=0) + np.min(point_cloud, axis=0)) / 2

    w = np.max(point_cloud, axis=0)[0] - np.min(point_cloud, axis=0)[0]
    h = np.max(point_cloud, axis=0)[1] - np.min(point_cloud, axis=0)[1]
    image_height = int(h / w * image_width) + 1

    point_cloud = point_cloud - mean

    print('Mean:', mean)
    print('Max:', np.max(point_cloud, axis=0))
    print('Min:', np.min(point_cloud, axis=0))
    print('w =', w, ', h =', h)
    print('n_w =', image_width, ', n_h =', image_height)
    print('Ratio:', w / h)

    D = np.ones((image_height, image_width), dtype=np.float) * np.max(depth)
    result_labels = np.ones((image_height, image_width)) * -1

    idx_sort = np.argsort(depth)

    C = np.empty_like(point_cloud, dtype=int)
    C[:, 0] = np.floor((point_cloud[:, 0] / h + 0.5) * image_height)
    C[:, 1] = np.floor((point_cloud[:, 1] / w + 0.5) * image_width)

    c1 = np.greater_equal(C[:, 0], 0)
    c2 = np.greater_equal(C[:, 1], 0)
    c3 = np.less(C[:, 0], image_height)
    c4 = np.less(C[:, 1], image_width)
    c = np.logical_and(np.logical_and(np.logical_and(c1, c2), c3), c4)

    # for i in range(depth.shape[0]):
    for i in range(depth.shape[0] - 1, -1, -1):
        j = idx_sort[i]
        if c[j]:
            D[C[j, 0], C[j, 1]] = depth[j]
            result_labels[C[j, 0], C[j, 1]] = labels[j]

    return D, result_labels


def project_to_depth_image_orthogonal_by_resultion(point_cloud, resolution, z_flip, img_size = None, label_map = -1):
    input_min = np.min(point_cloud, 0)
    input_max = np.max(point_cloud, 0)
    input_span = input_max - input_min
    image_width = math.ceil(float(input_span[0]) / float(resolution))
    point_cloud = np.array(point_cloud)
    point_cloud[:, 0:3] = point_cloud[:, 0:3]/resolution

    return project_to_depth_image_orthogonal(point_cloud, image_width, z_flip, img_size, label_map)


def project_to_depth_image_orthogonal(point_cloud, image_width, z_flip, img_size = None, label_map = -1):
    """
            Projects a point cloud orthogonal along the z-axis
            If there are multiple points per pixel the "highest depth value" is used
            If there are pixels without any points the "lowest value of the whole map" is used for them
            Attention: Currently all points are mapped to the output image.
                Meaning that the output image is

            :param point_cloud: (n_points x 3) np array
            :param image_width: desired with of output image. The height will automatically be calculated according
                to the aspect ratio of the input
            :param z_flip: weather the z-axis should be flipped around ( "up and down" )
            :returns [w_image,h_image] np array
        """

    calculate_real_labels = point_cloud.shape[1] > 3
    #print('calculate real labels: ' + str(calculate_real_labels))
    # Map x/y coordinates to discrete space
    input_min = np.min(point_cloud, 0)
    input_max = np.max(point_cloud, 0)
    input_span = input_max - input_min

    # "Clean data" to be positive
    point_cloud_centered = np.array(point_cloud)
    point_cloud_centered[:, 0:3] = point_cloud[:, 0:3] - input_min[0:3]

    # Flip z if needed
    input_z_scale = input_max[2] - input_min[2]
    if z_flip:
        point_cloud_centered[:, 2] = input_z_scale - point_cloud_centered[:, 2]

    if img_size == None:
        aspect_ratio = input_span[0] / input_span[1]
        image_height = int(image_width / aspect_ratio)
        image_size = [image_width, image_height]
    else:
        image_size = img_size
    #image_size = img_size
    # TODO Add comments - is input_span[2] as default value for "non-points" correct?
    if z_flip:
        result_depthmap = np.ones(image_size) * input_span[2]
    else:
        #result_depthmap = np.ones(image_size) * input_span[2]
        result_depthmap = np.zeros(image_size)
    result_labels = np.ones(image_size) * label_map

    # "project data" according to image_size
    # First get the desired output scale
    pixel_size = input_span[0:2] / image_size
    # Second recalculate coordinate of each point
    for p in point_cloud_centered:

        x = int(p[0] / pixel_size[0])
        if x == image_size[0]:
            x = image_size[0]-1

        y = int(p[1] / pixel_size[1])
        if y == image_size[1]:
            y = image_size[1]-1

        # save "highest" pixel in depth image
        if z_flip:
            if result_depthmap[x, y] > p[2]:
                result_depthmap[x, y] = p[2]
                if calculate_real_labels:
                    result_labels[x, y] = p[3]
                    #print('setting {0}/{1} with label {2}'.format(x, y, p[3]))
        else:
            #if p[2] > result_depthmap[x,y]:
            result_depthmap[x,y] = p[2]
            if calculate_real_labels:
                result_labels[x,y] = p[3]

    print("result_depthmap of orthogonal projection values range from " + str(np.min(result_depthmap)) + " to " + str(np.max(result_depthmap)))
    return result_depthmap, result_labels

def filter_point_cloud_by_labels(point_cloud, labels):

    result_workpiece = []
    result_non_workpiece = []

    input_min = np.min(point_cloud, 0)
    input_max = np.max(point_cloud, 0)
    input_span = input_max - input_min

    image_size = np.array([labels.shape[0], labels.shape[1]])
    pixel_size = input_span[0:2] / image_size

    for p in point_cloud:

        # get coordinates in projection space
        x = int(np.floor((p[0] - input_min[0])/ pixel_size[0]))
        if x >= image_size[0]:
            x = image_size[0]-1
        y = int(np.floor((p[1] - input_min[1]) / pixel_size[1]))
        if y >= image_size[1]:
            y = image_size[1] - 1

        # get label
        label = labels[x, y]

        # append the point to the corresponding result list
        if label == 0:
            result_non_workpiece.append(p)
        else:
            result_workpiece.append(p)

    return np.array(result_workpiece), np.array(result_non_workpiece)


def map_labels_to_point_cloud(point_cloud,labels):

    input_min = np.min(point_cloud,0)
    input_max = np.max(point_cloud,0)
    input_span = input_max - input_min

    image_size = np.array([labels.shape[0],labels.shape[1]])
    pixel_size = input_span[0:2]/image_size

    for ind in range(point_cloud.shape[0]):

        x = int(point_cloud[ind][0] - input_min[0]/pixel_size[0])
        if x == image_size[0]:
            x = image_size[0]-1
        y = int(point_cloud[ind][1]-input_min[1]/pixel_size[1])
        if y == image_size[1]:
            y = image_size[1]-1

        if x > image_size[0]:
            x = image_size[0]-2
            print("Check x")
        if y > image_size[1]:
            y = image_size[1]-2
            print("Check y")

        if not labels[x,y]==-1.0:
            point_cloud[ind][3]=labels[x,y]

    return point_cloud

def crop_point_cloud(point_cloud, lower_limit_x = 100000, upper_limit_x=-100000, lower_limit_y=100000, upper_limit_y=-100000, lower_limit_z=100000, upper_limit_z=-100000):
    cropped_point_cloud = []
    for p in point_cloud:
        if p[0] > lower_limit_x and p[0] < upper_limit_x and p[1] > lower_limit_y and p[1] < upper_limit_y and p[2] > lower_limit_z and p[2] < upper_limit_z:
            cropped_point_cloud.append(p)
    return np.asanyarray(cropped_point_cloud)