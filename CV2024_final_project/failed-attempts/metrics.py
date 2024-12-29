import cv2

def laplacian_variance_metric(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()