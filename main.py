import cv2 as cv
import matplotlib.pyplot as plt
import numpy
import numpy as np
from cv2.cv2 import SIFT, BFMatcher

from src.utils.files import read_images


def compute_keypoints_and_descriptors(image: np.array):
    sift: SIFT = cv.SIFT_create(nOctaveLayers=6)
    keypoints, descriptors = sift.detectAndCompute(image=image, mask=None)
    descriptors = descriptors.astype(numpy.uint8)

    return keypoints, descriptors


def match_descriptors(query_descriptors, train_descriptors, best_number: int = 20):
    bf: BFMatcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    matches = bf.match(
        queryDescriptors=query_descriptors, trainDescriptors=train_descriptors
    )
    return sorted(matches, key=lambda x: x.distance)[:best_number]


def get_keypoint_coords(keypoints_1, keypoints_2, matches):
    img1_coord = []
    img2_coord = []

    for m in matches:
        img1_coord.append(keypoints_1[m.queryIdx].pt)
        img2_coord.append(keypoints_2[m.trainIdx].pt)

    return img1_coord, img2_coord


def combine_images(image_1, image_2, matrix):
    dim_x = image_1.shape[1] + image_2.shape[1]
    dim_y = max(image_1.shape[0], image_2.shape[0])
    dim = (dim_x, dim_y)

    warped = cv.warpPerspective(image_2, matrix, dim)
    comb = warped.copy()
    comb[0 : image_1.shape[0], 0 : image_2.shape[1]] = image_1
    return comb


def connect_2_images(image_1, image_2):
    kp_1, des1 = compute_keypoints_and_descriptors(image_1)
    kp_2, des2 = compute_keypoints_and_descriptors(image_2)
    matches = match_descriptors(des1, des2)

    img1_pts, img2_pts = get_keypoint_coords(
        keypoints_1=kp_1, keypoints_2=kp_2, matches=matches
    )

    matrix, _ = cv.findHomography(
        srcPoints=np.float32(img2_pts), dstPoints=np.float32(img1_pts)
    )
    comb = combine_images(image_1, image_2, matrix)

    return comb


def main():
    images = read_images(images_dir="data/adobe_panoramas/data/rio")
    image_now = images[0]

    for image in images[1:]:
        comb = connect_2_images(image_1=image_now, image_2=image)
        image_now = image
        plt.imshow(comb)
        plt.show()


if __name__ == "__main__":
    main()
