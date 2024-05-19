import os
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage.transform import radon

def mean_filter(image, kernel_size1, kernel_size2):

    return cv2.blur(image, (kernel_size1, kernel_size2))

def get_Affine_transform(src_points, dst_points):
        # Calculate the affine transformation matrix
        affine_matrix = cv2.getAffineTransform(src_points, dst_points)
        height, width = affine_matrix.shape

        # padding affine matrix to be square
        if (height != width):
            affine_matrix = np.vstack([affine_matrix, [0, 0, 1]])

        return affine_matrix

def get_projective_transform(src_points, dst_points):
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    return perspective_matrix

def get_transform(matches, is_affine):

    src_points, dst_points = matches[:, 0], matches[:, 1]

    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    if(is_affine):
        transform_image = get_Affine_transform(src_points, dst_points)
    else:
        transform_image = get_projective_transform(src_points, dst_points)

    return transform_image

def inverse_transform_target_image(target_img, original_transform, canvas_size):

    # Calculate the inverse transformation matrix
    inverse_transform_matrix = np.linalg.inv(original_transform)

    # Perform inverse transformation using warpPerspective
    inverse_transformed_image = cv2.warpPerspective(target_img, inverse_transform_matrix, (canvas_size[1], canvas_size[0]))
    return inverse_transformed_image

def clean_baby(im):

    filteredIm = cv2.medianBlur(im, 3)
    canvas_height, canvas_width = im.shape
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    canvas_size = (canvas_height, canvas_width)
    for i in range(1,4):
        order = 3
        is_affine= True
        matches_data = os.path.join('Images', f"matches{i}.txt")
        if(i == 3):
            order = 4
            is_affine = False
        matches = np.loadtxt(matches_data, dtype=np.int64).reshape(1, order, 2, 2)
        original_transform = get_transform(matches[0], is_affine)
        inverse_transform_image = inverse_transform_target_image(filteredIm, original_transform, canvas_size)
        stitched_image = cv2.max(canvas, inverse_transform_image)
        fixed_image = +  stitched_image
        fixed_image = fixed_image
        cv2.imwrite(os.path.join('images', f"fixed{i}.jpg"), stitched_image)
        fixed_image = cv2.medianBlur(fixed_image, 3)

    return fixed_image

def clean_windmill(im):
    im_transform = fftshift(fft2(im))
    im_transform[124, 100] = 0
    im_transform[132, 156] = 0
    fixed_image = np.abs(ifft2(fftshift(im_transform)))

    return fixed_image


def clean_watermelon(im):

    # Define the sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])

    # Apply the kernel to the image
    sharpened_image = cv2.filter2D(im, -1, 1 * kernel)
    return sharpened_image


def clean_umbrella(im):
    F_A = np.fft.fft2(im)

    x0 = 79
    y0 = 4

    # Create a 2D Dirac delta function shifted by (x0, y0)
    shifted_delta = np.zeros_like(im)
    shifted_delta[y0, x0] = 1
    transform_shifted_delta = np.fft.fft2(shifted_delta)

    # Create a 2D Dirac delta function
    delta = np.zeros_like(im)
    delta[0, 0] = 1
    transform_delta = np.fft.fft2(delta) + 0.03

    # Combine the Fourier transforms of the shifted delta function and the original delta function
    denominator = transform_delta + transform_shifted_delta

    # Compute the inverse Fourier transform to recover the original image
    original_image_spectrum =2 * F_A / denominator
    original_image = np.fft.ifft2(original_image_spectrum).real

    return original_image


def clean_USAflag(im):
    # Calculate the vertical center
    height, width = im.shape
    center_height = height // 2

    # Separate into upper and lower parts
    upper_part = im[:center_height, :]
    lower_part = im[center_height:, :]
    filtered_im = cv2.medianBlur(lower_part, 5)
    filtered_im = mean_filter(filtered_im, 20,1)

    center_width = width // 2

    # Separate into left and right parts
    left_part = upper_part[:, :center_width]
    right_part = upper_part[:, center_width:]

    right_filtered = cv2.medianBlur(right_part, 5)
    right_filtered= mean_filter(right_filtered, 20,1)

    upper_part = np.concatenate((left_part, right_filtered), axis=1)
    full_image = np.concatenate((upper_part, filtered_im), axis=0)

    return full_image


def clean_house(im):
    F_A = np.fft.fft2(im)

    # Create a 2D Dirac delta function
    delta = np.zeros_like(im)
    for index in range(0,10):
        delta[0, index] = 1

    transform_delta = np.fft.fft2(delta)

    # Combine the Fourier transforms of the shifted delta function and the original delta function
    denominator = transform_delta

    # Compute the inverse Fourier transform to recover the original image
    original_image_spectrum =10 * F_A / denominator
    original_image = np.fft.ifft2(original_image_spectrum).real

    return original_image

def clean_bears(im):

    min_intensity = np.min(im)
    max_intensity = np.max(im)
    stretched_image = np.uint8((im - min_intensity) / (max_intensity - min_intensity) * 255)

    return stretched_image



