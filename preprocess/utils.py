"""helper function of pathology model"""

import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as et

import openslide
import tifffile
import cv2
import staintools


# ==================================================
# generate mask

def xml2mask(fn_wsi, fn_xml, fn_mask=None, mask_level=2):
    """from tumor XML to mask"""
    div = 4 ** mask_level

    img = openslide.OpenSlide(fn_wsi)
    shape = img.level_dimensions[mask_level]

    board_pos = np.zeros(shape[::-1], dtype=np.uint8)
    board_neg = np.zeros(shape[::-1], dtype=np.uint8)
    list_contour_pos = []
    list_contour_neg = []

    root = et.parse(fn_xml).getroot()
    regions = root.findall('Annotation')[0].findall('Regions')[0]

    for region in regions.findall('Region'):
        contour_pos = []
        contour_neg = []

        negative_flag = int(region.get('NegativeROA'))
        assert negative_flag == 0 or negative_flag == 1
        negative_flag = bool(negative_flag)

        list_vertex = region.findall('Vertices')[0].findall('Vertex')
        # last dot should be linked to the first dot
        list_vertex.append(list_vertex[0])

        for v in list_vertex:
            x = int(v.get('X').split('.')[0])
            y = int(v.get('Y').split('.')[0])
            x //= div
            y //= div

            if x >= shape[0]:
                x = shape[0] - 1
            elif x < 0:
                x = 0

            if y >= shape[1]:
                y = shape[1] - 1
            elif y < 0:
                y = 0

            if negative_flag:
                contour_neg.append((x, y))
            else:
                contour_pos.append((x, y))

        if contour_pos:
            list_contour_pos.append(np.array(contour_pos))
        else:
            list_contour_neg.append(np.array(contour_neg))

    board_pos = cv2.drawContours(board_pos, list_contour_pos, -1, [255, 0, 0], -1)
    board_neg = cv2.drawContours(board_neg, list_contour_neg, -1, [255, 0, 0], -1)
    mask = (board_pos > 0) * (board_neg == 0)
    mask = mask.astype(np.uint8) * 255

    if fn_mask is not None:
        tifffile.imsave(fn_mask, mask, compress=9)

    return mask


# ==================================================
# tessellate WSI to patches

def filter_patch(patch_array, num_small):
    """256 * 256 patch
    k*k is number of small patches
    """
    # G channel
    g_array = patch_array[:, :, 1]
    cnt_blank = 0
    small_size = g_array.shape[0] // num_small

    for i in range(num_small):
        for j in range(num_small):
            small = g_array[
                    i * small_size: (i+1) * small_size,
                    j * small_size: (j+1) * small_size]
            if small.mean() > 200 and small.std() < 20:
                cnt_blank += 1

    ratio_blank = cnt_blank / num_small ** 2

    return ratio_blank


def tessellate(
        fn_wsi, fn_mask, outdir, wsi,
        patch_level=1,
        patch_size=256,
        tumor_threshold=0.9,
        blank_threshold=0.5,
        num_small=8,
        color_normalizer=None,
        save_patch=True,
        test_one=False):
    """cut WSI image to small patches"""
    
    print('WSI: {}'.format(fn_wsi))

    mask_level = 2

    img = openslide.OpenSlide(fn_wsi)
    # shape: (y, x)
    mask = tifffile.imread(fn_mask)

    # number of patches
    xn_patch = img.level_dimensions[patch_level][0] // patch_size
    yn_patch = img.level_dimensions[patch_level][1] // patch_size

    mask_step = int(patch_size * (4 ** (patch_level - mask_level)))
    step = int(patch_size * (4 ** patch_level))

    cnt_patch = 0
    sr_list = []

    if save_patch:
        os.makedirs(outdir, exist_ok=True)

    for i in range(xn_patch):
        for j in range(yn_patch):
            # shape (y, x)
            mask_patch = mask[j * mask_step: (j + 1) * mask_step, i * mask_step: (i + 1) * mask_step]

            # filter patches with small tumor area
            tumor_ratio = (mask_patch > 0).mean()
            if tumor_ratio < tumor_threshold:
                continue

            # extract from WSI at raw resolution
            id_patch = '{}_l{}_{}_{}'.format(wsi, patch_level, i * step, j * step)

            patch_img = img.read_region(
                (i * step, j * step),
                patch_level,
                (patch_size, patch_size))

            # filter patches with large blank area
            patch_array = np.array(patch_img)[:, :, :3]
            blank_ratio = filter_patch(patch_array, num_small)

            if blank_ratio > blank_threshold:
                continue

            # statistics
            sr = pd.Series({
                'wsi': wsi,
                'patch': id_patch,
                'level': patch_level,
                'x': i * step,
                'y': j * step,
                'size': patch_size,
                'tumor_ratio': tumor_ratio,
                'blank_ratio': blank_ratio
            })

            sr_list.append(sr)
            cnt_patch += 1

            # color normalization
            if color_normalizer is not None:
                patch_array = color_normalizer.transform(patch_array)

            # save patch image
            if save_patch:
                fn_patch = os.path.join(
                    outdir,
                    '{}_l{}_{}_{}.tif'.format(wsi, patch_level, i * step, j * step))
                tifffile.imsave(fn_patch, patch_array, compress=9)

            if test_one:
                df_stat = pd.concat(sr_list, axis=1).T
                return df_stat

    df_stat = pd.concat(sr_list, axis=1).T
    print('patch level: {}, patch size: {}, {} / {} * {} patches'.format(
        patch_level, patch_size, cnt_patch, xn_patch, yn_patch))

    return df_stat


def overlap_tessellate(
        fn_wsi, fn_mask, outdir, wsi,
        patch_level=1,
        patch_size=256,
        tumor_threshold=0.9,
        blank_threshold=0.5,
        num_small=8,
        color_normalizer=None,
        save_patch=True,
        test_one=False):
    """cut WSI image to small patches"""

    print('WSI: {}'.format(fn_wsi))

    mask_level = 2

    img = openslide.OpenSlide(fn_wsi)
    # shape: (y, x)
    mask = tifffile.imread(fn_mask)

    img_xmax = img.level_dimensions[0][0]
    img_ymax = img.level_dimensions[0][1]
    img_step = int(patch_size * (4 ** patch_level))

    mask_step = int(patch_size * (4 ** (patch_level - mask_level)))
    mask_zoom = 4 ** mask_level

    cnt_patch = 0
    sr_list = []

    if save_patch:
        os.makedirs(outdir, exist_ok=True)

    for i in range(0, img_xmax - img_step, img_step // 2):
        for j in range(0, img_ymax - img_step, img_step // 2):
            # shape (y, x)
            mask_patch = mask[j // mask_zoom: j // mask_zoom + mask_step, i // mask_zoom: i // mask_zoom + mask_step]

            # filter patches with small tumor area
            tumor_ratio = (mask_patch > 0).mean()
            if tumor_ratio < tumor_threshold:
                continue

            # extract from WSI at raw resolution
            id_patch = '{}_l{}_{}_{}'.format(wsi, patch_level, i, j)

            patch_img = img.read_region(
                (i, j),
                patch_level,
                (patch_size, patch_size))

            # filter patches with large blank area
            patch_array = np.array(patch_img)[:, :, :3]
            blank_ratio = filter_patch(patch_array, num_small)

            if blank_ratio > blank_threshold:
                continue

            # statistics
            sr = pd.Series({
                'wsi': wsi,
                'patch': id_patch,
                'level': patch_level,
                'x': i,
                'y': j,
                'size': patch_size,
                'tumor_ratio': tumor_ratio,
                'blank_ratio': blank_ratio
            })

            sr_list.append(sr)
            cnt_patch += 1

            # color normalization
            if color_normalizer is not None:
                patch_array = color_normalizer.transform(patch_array)

            # save patch image
            if save_patch:
                fn_patch = os.path.join(
                    outdir,
                    '{}_l{}_{}_{}.tif'.format(wsi, patch_level, i, j))
                tifffile.imsave(fn_patch, patch_array, compress=9)

            if test_one:
                df_stat = pd.concat(sr_list, axis=1).T
                return df_stat

    df_stat = pd.concat(sr_list, axis=1).T
    print('patch level: {}, patch size: {}, patch number: {}'.format(
        patch_level, patch_size, cnt_patch))

    return df_stat


# ===================================================
# color normalizer

class ColorNormalizer(object):
    def __init__(self, target_fname, method):
        target = staintools.read_image(target_fname)
        self.normalizer = staintools.StainNormalizer(method=method)
        self.normalizer.fit(target)

    def transform(self, img=None, fname=None):
        if fname is not None:
            img = staintools.read_image(fname)

        normalized_img = self.normalizer.transform(img)

        return normalized_img


# ===================================================
# assemble patches to WSI

def assemble_patch(patch_dir, patch_level=1, patch_size=256, xmax=None, ymax=None):
    """assemble patches
    filename of patch: training_data_01_l1_49152_69632.tif
    training_data_LEVEL_X_Y.tif
    """
    zoom = 4 ** patch_level

    if xmax is None or ymax is None:
        list_x = [int(fn_base.split('.')[0].split('_')[4]) for fn_base in os.listdir(patch_dir)]
        list_y = [int(fn_base.split('.')[0].split('_')[5]) for fn_base in os.listdir(patch_dir)]
        range_x = (min(list_x), max(list_x))
        range_y = (min(list_y), max(list_y))
    else:
        range_x = (0, xmax)
        range_y = (0, ymax)

    # shape: (y, x, RGB)
    img = np.zeros(((range_y[1] - range_y[0]) // zoom + patch_size,
                    (range_x[1] - range_x[0]) // zoom + patch_size,
                    3))

    cnt = 0

    for fn_base in os.listdir(patch_dir):
        fn = os.path.join(patch_dir, fn_base)
        patch = tifffile.imread(fn)
        patch = patch[:, :, :3]

        x = int(fn_base.split('.')[0].split('_')[4])
        y = int(fn_base.split('.')[0].split('_')[5])

        xnew = (x - range_x[0]) // zoom
        ynew = (y - range_y[0]) // zoom

        if ynew + patch_size <= img.shape[0] and xnew + patch_size <= img.shape[1]:
            img[ynew: (ynew + patch_size), xnew: (xnew + patch_size), :] = patch
            cnt += 1
        else:
            print(xnew, ynew, img.shape)

    print(cnt)

    return img


# =======================================================
# create xml given mask vertices

def generate_box(x, y, size):
    """"generate boundary given left-top coordinate"""
    list_xy = [
        (x, y),
        (x + size, y),
        (x + size, y + size),
        (x, y + size),
        (x, y)
    ]

    return list_xy


def create_xml(in_xml, out_xml, x, y, size):
    """add vertex to xml"""
    list_xy = generate_box(x, y, size)

    tree = et.parse(in_xml)
    for vertices in tree.iter('Vertices'):
        for x, y in list_xy:
            et.SubElement(vertices, 'Vertex', {'Z': '0', 'Y': str(y), 'X': str(x)})

    tree.write(out_xml)


