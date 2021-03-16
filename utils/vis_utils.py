import cv2


def draw_rect(img, bbox_points, color=(0, 255, 0), thickness=3, is_normalized=False):
    """ Draw rectangle
    Args:
        img: image
        bbox_points: [xmin, ymin, xmax, ymax]
        color: color rgb value
        thickness: line thickness
        is_normalized: Normalized points or not
    Return:
        img
    """
    if is_normalized:
        h, w = img.shape[:2]
        xmin = int(bbox_points[0] * w)
        ymin = int(bbox_points[1] * h)
        xmax = int(bbox_points[2] * w)
        ymax = int(bbox_points[3] * h)
    else:
        xmin = int(bbox_points[0])
        ymin = int(bbox_points[1])
        xmax = int(bbox_points[2])
        ymax = int(bbox_points[3])
    
    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
    return img


def draw_marks(img, points_2d, color=(0, 255, 0), thickness=3, is_normalized=True):
    if is_normalized:
        h, w = img.shape[:2]
        for (x, y) in points_2d:
            cv2.circle(img, (int(x * w), int(y * h)), 1, color, thickness, 1)
    
    else:
        for (x, y) in points_2d:
            cv2.circle(img, (int(x), int(y)), 1, color, thickness, 1)
