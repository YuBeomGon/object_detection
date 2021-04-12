import cv2
import numpy as np
import albumentations as A

transforms = A.Compose([
    A.CenterCrop(1350,1350, True,1),
], p=1.0, bbox_params=A.BboxParams(format='pascal_voc', min_area=0, min_visibility=0.5, label_fields=['labels'])) 

def switch_image(img) :
    h, w = img.shape[:2]
    if (h, w) == (4032, 1960) or (h, w) == (4000, 1800) :
        img = np.flip(img, 1)
        img = np.transpose(img, (1, 0, 2))      
    return img

# def crop_image(img):
#     h, w = img.shape[:2]
#     # Case 1
#     if (h, w) == (1960, 4032):
#         h_margin = 305
#         w_margin = 1341

#     elif (h, w) == (1800, 4000):
#         h_margin = 225
#         w_margin = 1325
    
#     else: #(1560, 1632)
#         h_margin = 0
#         w_margin = 0

#     wid = w - w_margin*2
#     hgt = h - h_margin*2
#     img = img[h_margin:-h_margin, w_margin:-w_margin, :]
    
#     return img


# def transform_bbox_points(img, bbox_point):
#     h, w = img.shape[:2]
#     # Case 1
#     if (h, w) == (1960, 4032):
#         h_margin = 305
#         w_margin = 1341

#     elif (h, w) == (1800, 4000):
#         h_margin = 225
#         w_margin = 1325
    
#     else: #(1560, 1632)
#         h_margin = 0
#         w_margin = 0
    
#     xmin, ymin, xmax, ymax = bbox_point
#     xmin -= w_margin
#     ymin -= h_margin
#     xmax -= w_margin
#     ymax -= h_margin
#     new_bbox_point = [xmin, ymin, xmax, ymax]
    
#     return new_bbox_point


def shift(img, val_x, val_y, points_2d=None, is_normalized=True):
    """ Shift Image and Points
    """
    h, w = img.shape[:2]
    shift_x = int(val_x * w)
    shift_y = int(val_y * h)

    # Get Affine transform matrix
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    
    # For image
    img = cv2.warpAffine(img, M, (w, h))
    
    if points_2d is not None:
        # For 2d points
        points_2d = np.array(points_2d).reshape((-1, 2))
        
        if is_normalized:
            points_2d[:, 0] *= w
            points_2d[:, 1] *= h
        
        ones = np.ones(shape=(len(points_2d), 1))
        points_ones = np.hstack([points_2d, ones])
        transformed_points = np.empty_like(points_2d)
        for idx, point in enumerate(points_ones):
            point = np.dot(M, point)
            transformed_points[idx,] = point
        
        if is_normalized:
            transformed_points[:, 0] /= w
            transformed_points[:, 1] /= h
        return img, transformed_points
    else:
        return img


def rotate(img, value, points_2d=None, is_normalized=True):
    h, w = img.shape[:2]
    if points_2d is not None:
        points_2d = np.array(points_2d).reshape((-1, 2))
        if is_normalized:
            points_2d[:, 0] *= w
            points_2d[:, 1] *= h
        
        c_x = np.mean(points_2d[:, 0])
        c_y = np.mean(points_2d[:, 1])

        # Get Affine transform matrix
        M = cv2.getRotationMatrix2D((c_x, c_y), value, 1.0)

        # For image
        img = cv2.warpAffine(img, M, (w, h))
        # For 2d points
        ones = np.ones(shape=(len(points_2d), 1))
        points_ones = np.hstack([points_2d, ones])
        transformed_points = np.empty_like(points_2d)
        for idx, point in enumerate(points_ones):
            point = np.dot(M, point)
            transformed_points[idx,] = point
        
        if is_normalized:
            transformed_points[:, 0] /= w
            transformed_points[:, 1] /= h
        return img, transformed_points
    
    else:
        c_x = w / 2
        c_y = h / 2
        # Get Affine transform matrix
        M = cv2.getRotationMatrix2D((c_x, c_y), value, 1.0)
        # For image
        img = cv2.warpAffine(img, M, (w, h))
        return img
