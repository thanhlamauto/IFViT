# utils_jax.py
# Ported from FingerNet/src/utils.py for JAX/Flax inference
# Python 3 compatible version

import os
import sys
import glob
import shutil
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal, spatial, sparse
import cv2


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def re_mkdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def init_log(output_dir):
    re_mkdir(output_dir)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(message)s',
        datefmt='%Y%m%d-%H:%M:%S',
        filename=os.path.join(output_dir, 'log.log'),
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging


def copy_file(path_s, path_t):
    shutil.copy(path_s, path_t)


def get_files_in_folder(folder, file_ext=None):
    files = glob.glob(os.path.join(folder, "*" + file_ext))
    files_name = []
    for i in files:
        _, name = os.path.split(i)
        name, ext = os.path.splitext(name)
        files_name.append(name)
    return np.asarray(files), np.asarray(files_name)


def point_rot(points, theta, b_size, a_size):
    cosA = np.cos(theta)
    sinA = np.sin(theta)
    b_center = [b_size[1] / 2.0, b_size[0] / 2.0]
    a_center = [a_size[1] / 2.0, a_size[0] / 2.0]
    points = np.dot(points - b_center, np.array([[cosA, -sinA], [sinA, cosA]])) + a_center
    return points


def mnt_reader(file_name):
    minutiae = []
    with open(file_name) as f:
        for i, line in enumerate(f):
            if i < 2 or len(line.strip()) == 0:
                continue
            w, h, o = [float(x) for x in line.split()]
            w, h = int(round(w)), int(round(h))
            minutiae.append([w, h, o])
    return minutiae


def mnt_writer(mnt, image_name, image_size, file_name):
    with open(file_name, 'w') as f:
        f.write(f'{image_name}\n')
        f.write(f'{mnt.shape[0]} {image_size[0]} {image_size[1]}\n')
        for i in range(mnt.shape[0]):
            f.write('%d %d %.6f\n' % (mnt[i, 0], mnt[i, 1], mnt[i, 2]))


def gabor_fn(ksize, sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma
    xmax = ksize[0] / 2
    ymax = ksize[1] / 2
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gb_cos = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * \
             np.cos(2 * np.pi / Lambda * x_theta + psi)
    gb_sin = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * \
             np.sin(2 * np.pi / Lambda * x_theta + psi)
    return gb_cos, gb_sin


def gabor_bank(stride=2, Lambda=8):
    num_k = int(180 / stride)
    filters_cos = np.ones([25, 25, num_k], dtype=float)
    filters_sin = np.ones([25, 25, num_k], dtype=float)
    for n, i in enumerate(range(-90, 90, stride)):
        theta = i * np.pi / 180.0
        kernel_cos, kernel_sin = gabor_fn((24, 24), 4.5, -theta, Lambda, 0, 0.5)
        filters_cos[..., n] = kernel_cos
        filters_sin[..., n] = kernel_sin
    filters_cos = np.reshape(filters_cos, [25, 25, 1, -1])
    filters_sin = np.reshape(filters_sin, [25, 25, 1, -1])
    return filters_cos, filters_sin


def gaussian2d(shape=(5, 5), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def gausslabel(length=180, stride=2):
    # scipy.signal.gaussian was removed, use manual implementation
    # Equivalent to: signal.gaussian(length + 1, std=3)
    x = np.arange(length + 1)
    center = length / 2.0
    std = 3.0
    gaussian_pdf = np.exp(-0.5 * ((x - center) / std) ** 2)
    gaussian_pdf = gaussian_pdf / gaussian_pdf.sum()  # normalize
    
    label = np.reshape(np.arange(stride / 2, length, stride), [1, 1, -1, 1])
    y = np.reshape(np.arange(stride / 2, length, stride), [1, 1, 1, -1])
    delta = np.array(np.abs(label - y), dtype=float)
    delta = np.minimum(delta, length - delta) + length / 2
    # Clip to valid range and convert to int for indexing
    delta = np.clip(delta, 0, length).astype(int)
    return gaussian_pdf[delta]


def angle_delta(A, B, max_D=np.pi * 2):
    delta = np.abs(A - B)
    delta = np.minimum(delta, max_D - delta)
    return delta


def fmeasure(P, R):
    return 2 * P * R / (P + R + 1e-10)


def distance(y_true, y_pred, max_D=16, max_O=np.pi / 6):
    D = spatial.distance.cdist(y_true[:, :2], y_pred[:, :2], 'euclidean')
    O = spatial.distance.cdist(
        np.reshape(y_true[:, 2], [-1, 1]),
        np.reshape(y_pred[:, 2], [-1, 1]),
        angle_delta
    )
    return (D <= max_D) * (O <= max_O)


def mnt_P_R_F(y_true, y_pred, maxd=15, maxo=np.pi / 6):
    if y_pred.shape[0] == 0 or y_true.shape[0] == 0:
        return 0, 0, 0, 0, 0
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    total_gt, total = float(y_true.shape[0]), float(y_pred.shape[0])
    dis = spatial.distance.cdist(y_pred[:, :2], y_true[:, :2], 'euclidean')
    mindis, idx = dis.min(axis=1), dis.argmin(axis=1)
    angle = np.abs(np.mod(y_pred[:, 2], 2 * np.pi) - y_true[idx, 2])
    angle = np.asarray([angle, 2 * np.pi - angle]).min(axis=0)
    precision = len(np.unique(idx[(mindis <= maxd) & (angle <= maxo)])) / float(y_pred.shape[0])
    recall = len(np.unique(idx[(mindis <= maxd) & (angle <= maxo)])) / float(y_true.shape[0])
    if recall != 0:
        loc = np.mean(mindis[(mindis <= maxd) & (angle <= maxo)])
        ori = np.mean(angle[(mindis <= maxd) & (angle <= maxo)])
    else:
        loc = 0
        ori = 0
    return precision, recall, fmeasure(precision, recall), loc, ori


def nms(mnt):
    if mnt.shape[0] == 0:
        return mnt
    mnt_sort = mnt.tolist()
    mnt_sort.sort(key=lambda x: x[3], reverse=True)
    mnt_sort = np.array(mnt_sort)
    inrange = distance(mnt_sort, mnt_sort, max_D=16, max_O=np.pi / 6).astype(np.float32)
    keep_list = np.ones(mnt_sort.shape[0])
    for i in range(mnt_sort.shape[0]):
        if keep_list[i] == 0:
            continue
        keep_list[i + 1:] = keep_list[i + 1:] * (1 - inrange[i, i + 1:])
    return mnt_sort[keep_list.astype(bool), :]


def draw_minutiae(image, minutiae, fname, r=15):
    image = np.squeeze(image)
    fig = plt.figure()
    plt.imshow(image, cmap='gray')
    plt.plot(minutiae[:, 0], minutiae[:, 1], 'rs', fillstyle='none', linewidth=1)
    for x, y, o in minutiae:
        plt.plot([x, x + r * np.cos(o)], [y, y + r * np.sin(o)], 'r-')
    plt.axis([0, image.shape[1], image.shape[0], 0])
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def draw_ori_on_img(img, ori, mask, fname, coh=None, stride=16):
    ori = np.squeeze(ori)
    mask = np.squeeze(np.round(mask))
    img = np.squeeze(img)
    ori = ndimage.zoom(ori, np.array(img.shape) / np.array(ori.shape, dtype=float), order=0)
    if mask.shape != img.shape:
        mask = ndimage.zoom(mask, np.array(img.shape) / np.array(mask.shape, dtype=float), order=0)
    if coh is None:
        coh = np.ones_like(img)
    fig = plt.figure()
    plt.imshow(img, cmap='gray')
    for i in range(stride, img.shape[0], stride):
        for j in range(stride, img.shape[1], stride):
            if mask[i, j] == 0:
                continue
            x, y, o, r = j, i, ori[i, j], coh[i, j] * (stride * 0.9)
            plt.plot([x, x + r * np.cos(o)], [y, y + r * np.sin(o)], 'r-')
    plt.axis([0, img.shape[1], img.shape[0], 0])
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def label2mnt(mnt_s_out, mnt_w_out, mnt_h_out, mnt_o_out, thresh=0.5):
    """
    Convert model outputs to minutiae list.
    Ported from train_test_deploy.py label2mnt().
    """
    mnt_s_out = np.squeeze(mnt_s_out)
    mnt_w_out = np.squeeze(mnt_w_out)
    mnt_h_out = np.squeeze(mnt_h_out)
    mnt_o_out = np.squeeze(mnt_o_out)
    assert len(mnt_s_out.shape) == 2 and len(mnt_w_out.shape) == 3 and \
           len(mnt_h_out.shape) == 3 and len(mnt_o_out.shape) == 3

    mnt_sparse = sparse.coo_matrix(mnt_s_out > thresh)
    mnt_list = np.array(list(zip(mnt_sparse.row, mnt_sparse.col)), dtype=np.int32)
    if mnt_list.shape[0] == 0:
        return np.zeros((0, 4))

    mnt_w_out = np.argmax(mnt_w_out, axis=-1)
    mnt_h_out = np.argmax(mnt_h_out, axis=-1)
    mnt_o_out = np.argmax(mnt_o_out, axis=-1)

    mnt_final = np.zeros((len(mnt_list), 4))
    mnt_final[:, 0] = mnt_sparse.col * 8 + mnt_w_out[mnt_list[:, 0], mnt_list[:, 1]]
    mnt_final[:, 1] = mnt_sparse.row * 8 + mnt_h_out[mnt_list[:, 0], mnt_list[:, 1]]
    mnt_final[:, 2] = (mnt_o_out[mnt_list[:, 0], mnt_list[:, 1]] * 2 - 89.) / 180 * np.pi
    mnt_final[mnt_final[:, 2] < 0.0, 2] = mnt_final[mnt_final[:, 2] < 0.0, 2] + 2 * np.pi
    mnt_final[:, 3] = mnt_s_out[mnt_list[:, 0], mnt_list[:, 1]]
    return mnt_final
