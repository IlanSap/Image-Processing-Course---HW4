import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt

image_path = 'zebra.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Resize an image using Fourier transform
def resize_image_fourier(input_image):
    # Compute the Fourier Transform of the input image
    fourier_transform = fft2(input_image)

    # Shift the zero frequency component to the center
    shifted_fourier = fftshift(fourier_transform)

    # Pad the Fourier-transformed image to increase its size
    # Get the height and width of the input image
    h, w = input_image.shape
    # Define the scaling factor
    scale_factor = 4
    # Pad the Fourier-transformed image to increase its size
    padded_fourier = np.pad(shifted_fourier, ((h//2, h//2), (w//2, w//2)), mode='constant') * scale_factor

    # Compute the inverse Fourier Transform to get the resized image
    resized_image = np.abs(ifft2(ifftshift(padded_fourier)))

    # Compute the magnitude spectrum of the padded Fourier-transformed image
    magnitude_spectrum = np.log(1 + np.abs(padded_fourier))

    return magnitude_spectrum, resized_image


# Generate an image with four replicated copies of the input image
def four_replicated_images(input_image):
    # Create meshgrid for the indices
    u, v = np.meshgrid(np.arange(2 * input_image.shape[1]), np.arange(2 * input_image.shape[0]))

    # Compute the indices for replicating the input image
    indices = (v % input_image.shape[0], u % input_image.shape[1])

    # Create four replicated images by indexing the input image and scaling down by a factor of 4
    four_images = input_image[indices] / 4

    return four_images


plt.figure(figsize=(10, 10))

# Display the Original Grayscale Image
plt.subplot(321)
plt.title('Original Grayscale Image', fontsize=10)
plt.imshow(image, cmap='gray')
plt.axis('off')

# Display Fourier Spectrum of the Original Grayscale Image
plt.subplot(322)
plt.title('Fourier Spectrum', fontsize=10)
plt.imshow(np.log(1 + np.abs(fftshift(fft2(image)))), cmap='gray')
plt.axis('off')

# Display the Fourier Spectrum with Zero Padding
image_spectrum, resized_image_zero_padding = resize_image_fourier(image)
plt.subplot(323)
plt.title('Fourier Spectrum Zero Padding', fontsize=10)
plt.imshow(image_spectrum, cmap='gray')
plt.axis('off')

# Display Resized Images using Zero Padding
plt.subplot(324)
plt.title('Two Times Larger Grayscale Image', fontsize=10)
plt.imshow(resized_image_zero_padding, cmap='gray')
plt.axis('off')

# Display the Fourier Spectrum of Four Replicated Images
replicated_image = four_replicated_images(image)
plt.subplot(325)
plt.title('Fourier Spectrum Four Copies', fontsize=10)
plt.imshow(np.log(1 + np.abs(fftshift(fft2(replicated_image)))), cmap='gray')
plt.axis('off')

# Display the Four Replicated Images
plt.subplot(326)
plt.title('Four Copies Grayscale Image', fontsize=10)
plt.imshow(replicated_image, cmap='gray')
plt.axis('off')

plt.tight_layout()

plt.savefig('zebra_scaled.png')

plt.show()
