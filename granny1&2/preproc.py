import base64
from io import BytesIO
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications import vgg16, resnet50, inception_v3
import cv2
import ast


# Load MobileNetV2 model with ImageNet weights
model = tf.keras.applications.MobileNetV2(weights='imagenet')

preprocessors = [
    (lambda img: same(img), "Original"),
]

def load_labels(label_path):
    """
    Load labels from the given file and return a mapping of indices to label names.
    """
    with open(label_path, 'r') as file:
        lines = file.readlines()
        labels = {}
        for idx, line in enumerate(lines):
            names = [name.strip() for name in line.split(',')]
            labels[str(idx)] = names
        return labels


# Load the labels
labels = load_labels("labels.txt")


def same(img_bytes):
    """
    Return the same image bytes as provided in input.
    """
    with Image.open(BytesIO(img_bytes)) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return buffered.getvalue()


def resize_image(img_bytes, target_size=(224, 224), interpolation='bilinear'):
    with Image.open(BytesIO(img_bytes)) as img:
        if interpolation == 'bilinear':
            method = Image.BILINEAR
        elif interpolation == 'nearest':
            method = Image.NEAREST
        elif interpolation == 'bicubic':
            method = Image.BICUBIC
        else:
            raise ValueError("Unknown interpolation method")
        img_resized = img.resize(target_size, method)
        buffered = BytesIO()
        img_resized.save(buffered, format="PNG")
        return buffered.getvalue()


def rotate_image(img_bytes, angle=0):
    with Image.open(BytesIO(img_bytes)) as img:
        img_rotated = img.rotate(angle)
        buffered = BytesIO()
        img_rotated.save(buffered, format="PNG")
        return buffered.getvalue()


def adjust_brightness(img_bytes, factor=1.0):
    with Image.open(BytesIO(img_bytes)) as img:
        img_enhanced = ImageEnhance.Brightness(img).enhance(factor)
        buffered = BytesIO()
        img_enhanced.save(buffered, format="PNG")
        return buffered.getvalue()


def compress_jpeg(img_bytes, quality=95):
    with Image.open(BytesIO(img_bytes)) as img:
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=quality)
        return buffered.getvalue()

def compress_png(img_bytes, compression_level=0):
    with Image.open(BytesIO(img_bytes)) as img:
        buffered = BytesIO()
        img.save(buffered, format="PNG", compress_level=compression_level)
        return buffered.getvalue()

def compress_gif(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        buffered = BytesIO()
        img.save(buffered, format="GIF")
        return buffered.getvalue()

def reduce_color_depth(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        img_reduced = img.convert('P')
        buffered = BytesIO()
        img_reduced.save(buffered, format="PNG")
        return buffered.getvalue()

def crop_image(img_bytes, left=0, upper=0, right=224, lower=224):
    with Image.open(BytesIO(img_bytes)) as img:
        img_cropped = img.crop((left, upper, right, lower))
        buffered = BytesIO()
        img_cropped.save(buffered, format="PNG")
        return buffered.getvalue()

def gaussian_blur(img_bytes, radius=2):
    with Image.open(BytesIO(img_bytes)) as img:
        img_blurred = img.filter(ImageFilter.GaussianBlur(radius))
        buffered = BytesIO()
        img_blurred.save(buffered, format="PNG")
        return buffered.getvalue()

def median_blur(img_bytes, size=3):
    with Image.open(BytesIO(img_bytes)) as img:
        img_blurred = img.filter(ImageFilter.MedianFilter(size))
        buffered = BytesIO()
        img_blurred.save(buffered, format="PNG")
        return buffered.getvalue()

def adjust_contrast(img_bytes, factor=1.0):
    with Image.open(BytesIO(img_bytes)) as img:
        img_enhanced = ImageEnhance.Contrast(img).enhance(factor)
        buffered = BytesIO()
        img_enhanced.save(buffered, format="PNG")
        return buffered.getvalue()

def adjust_sharpness(img_bytes, factor=1.0):
    with Image.open(BytesIO(img_bytes)) as img:
        img_enhanced = ImageEnhance.Sharpness(img).enhance(factor)
        buffered = BytesIO()
        img_enhanced.save(buffered, format="PNG")
        return buffered.getvalue()

def convert_grayscale(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        img_gray = img.convert('L')
        buffered = BytesIO()
        img_gray.save(buffered, format="PNG")
        return buffered.getvalue()

def apply_sepia(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        width, height = img.size
        pixels = img.load()  # Load pixel data

        for py in range(height):
            for px in range(width):
                r, g, b = img.getpixel((px, py))

                tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                tb = int(0.272 * r + 0.534 * g + 0.131 * b)

                pixels[px, py] = (tr, tg, tb)

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return buffered.getvalue()

def binary_threshold(img_bytes, threshold=128):
    with Image.open(BytesIO(img_bytes)) as img:
        img_gray = img.convert('L')
        img_binary = img_gray.point(lambda p: p > threshold and 255)
        buffered = BytesIO()
        img_binary.save(buffered, format="PNG")
        return buffered.getvalue()

def convert_to_hsv(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        img_hsv = img.convert('HSV')
        buffered = BytesIO()
        img_hsv.save(buffered, format="PNG")
        return buffered.getvalue()

def convert_to_lab(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        img_lab = img.convert('LAB')
        buffered = BytesIO()
        img_lab.save(buffered, format="PNG")
        return buffered.getvalue()

def skew_image(img_bytes, direction='x', degree=0):
    with Image.open(BytesIO(img_bytes)) as img:
        if direction == 'x':
            transform_matrix = [1, degree, 0, 0, 1, 0]
        elif direction == 'y':
            transform_matrix = [1, 0, 0, degree, 1, 0]
        else:
            raise ValueError("Invalid direction. Choose 'x' or 'y'.")
        img_skewed = img.transform(img.size, Image.AFFINE, transform_matrix)
        buffered = BytesIO()
        img_skewed.save(buffered, format="PNG")
        return buffered.getvalue()

def min_max_scaling(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        np_img = np.array(img)
        scaled_img = 255 * (np_img - np_img.min()) / (np_img.max() - np_img.min())
        img_scaled = Image.fromarray(np.uint8(scaled_img))
        buffered = BytesIO()
        img_scaled.save(buffered, format="PNG")
        return buffered.getvalue()


def split_channel(img_bytes, channel='R'):
    with Image.open(BytesIO(img_bytes)) as img:
        r, g, b = img.split()
        if channel == 'R':
            channel_img = r
        elif channel == 'G':
            channel_img = g
        elif channel == 'B':
            channel_img = b
        buffered = BytesIO()
        channel_img.save(buffered, format="PNG")
        return buffered.getvalue()

def swap_channels(img_bytes, swap='RG'):
    with Image.open(BytesIO(img_bytes)) as img:
        r, g, b = img.split()
        if swap == 'RG':
            img = Image.merge('RGB', (g, r, b))
        elif swap == 'RB':
            img = Image.merge('RGB', (b, g, r))
        elif swap == 'GB':
            img = Image.merge('RGB', (r, b, g))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return buffered.getvalue()

def add_salt_pepper_noise(img_bytes, prob=0.05):
    with Image.open(BytesIO(img_bytes)) as img:
        np_img = np.array(img)
        output = np.zeros_like(np_img)
        thres = 1 - prob
        for i in range(np_img.shape[0]):
            for j in range(np_img.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = np_img[i][j]
        img_output = Image.fromarray(output)
        buffered = BytesIO()
        img_output.save(buffered, format="PNG")
        return buffered.getvalue()

def add_gaussian_noise(img_bytes, mean=0, sigma=25):
    with Image.open(BytesIO(img_bytes)) as img:
        np_img = np.array(img)
        gauss = np.random.normal(mean, sigma, np_img.shape)
        noisy = np.clip(np_img + gauss, 0, 255)
        img_noisy = Image.fromarray(np.uint8(noisy))
        buffered = BytesIO()
        img_noisy.save(buffered, format="PNG")
        return buffered.getvalue()

def histogram_equalization(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        np_img = np.array(img)
        img_yuv = cv2.cvtColor(np_img, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_output = Image.fromarray(cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB))
        buffered = BytesIO()
        img_output.save(buffered, format="PNG")
        return buffered.getvalue()

def unsharp_mask(img_bytes, radius=1, percent=150):
    with Image.open(BytesIO(img_bytes)) as img:
        blurred = img.filter(ImageFilter.GaussianBlur(radius))
        sharpened = Image.blend(img, blurred, percent / 100.0)
        buffered = BytesIO()
        sharpened.save(buffered, format="PNG")
        return buffered.getvalue()

def image_to_bytes(img):
    """Helper function to convert PIL Image to bytes."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return buffered.getvalue()

def horizontal_flip(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return image_to_bytes(flipped_img)

def vertical_flip(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return image_to_bytes(flipped_img)

def zoom_image(img_bytes, factor=1.5):
    with Image.open(BytesIO(img_bytes)) as img:
        x, y = img.size
        new_x, new_y = int(x/factor), int(y/factor)
        left, upper = (x - new_x) // 2, (y - new_y) // 2
        right, lower = left + new_x, upper + new_y
        img_cropped = img.crop((left, upper, right, lower))
        img_zoomed = img_cropped.resize((x, y))
        return image_to_bytes(img_zoomed)

def gamma_correction(img_bytes, gamma=1.0):
    with Image.open(BytesIO(img_bytes)) as img:
        img_np = np.array(img)
        img_corrected = np.clip(255 * (img_np / 255) ** gamma, 0, 255)
        corrected_img = Image.fromarray(img_corrected.astype(np.uint8))
        return image_to_bytes(corrected_img)

def negative_image(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        img_np = np.array(img)
        img_negative = 255 - img_np
        negative_img = Image.fromarray(img_negative)
        return image_to_bytes(negative_img)

def posterize(img_bytes, bits=4):
    with Image.open(BytesIO(img_bytes)) as img:
        levels = 2**bits
        img_np = np.array(img)
        img_posterized = np.clip(((img_np // levels) * levels + levels / 2), 0, 255)
        posterized_img = Image.fromarray(img_posterized.astype(np.uint8))
        return image_to_bytes(posterized_img)

def convert_to_hsv_corrected(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        img_np = np.array(img)
        img_hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return image_to_bytes(Image.fromarray(img_rgb))

def convert_to_lab_corrected(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        img_np = np.array(img)
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2Lab)
        img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_Lab2RGB)
        return image_to_bytes(Image.fromarray(img_rgb))

def otsu_binarization(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        img_gray = img.convert('L')
        threshold = cv2.threshold(np.array(img_gray), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        buffered = BytesIO()
        Image.fromarray(threshold).save(buffered, format="PNG")
        return buffered.getvalue()


def morphological_open(img_bytes, kernel_size=3):
    with Image.open(BytesIO(img_bytes)) as img:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened = cv2.morphologyEx(np.array(img), cv2.MORPH_OPEN, kernel)
        buffered = BytesIO()
        Image.fromarray(opened).save(buffered, format="PNG")
        return buffered.getvalue()

def morphological_close(img_bytes, kernel_size=3):
    with Image.open(BytesIO(img_bytes)) as img:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        closed = cv2.morphologyEx(np.array(img), cv2.MORPH_CLOSE, kernel)
        buffered = BytesIO()
        Image.fromarray(closed).save(buffered, format="PNG")
        return buffered.getvalue()

def log_transform(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        img_np = np.array(img).astype(np.float32)
        img_log_transformed = np.log1p(img_np)
        img_scaled = 255 * img_log_transformed / np.max(img_log_transformed)
        buffered = BytesIO()
        Image.fromarray(img_scaled.astype(np.uint8)).save(buffered, format="PNG")
        return buffered.getvalue()

def gamma_transform(img_bytes, gamma=1.0):
    with Image.open(BytesIO(img_bytes)) as img:
        img_np = np.array(img)/255.0
        img_power_transformed = np.power(img_np, gamma)
        buffered = BytesIO()
        Image.fromarray((img_power_transformed*255).astype(np.uint8)).save(buffered, format="PNG")
        return buffered.getvalue()

def bit_plane_slicing(img_bytes, plane=1):
    with Image.open(BytesIO(img_bytes)) as img:
        img_np = np.array(img)
        bit_sliced = (img_np & (1 << plane)) >> plane
        buffered = BytesIO()
        Image.fromarray((bit_sliced * 255).astype(np.uint8)).save(buffered, format="PNG")
        return buffered.getvalue()
def vgg_preprocessing(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        img_np = np.array(img.resize((224, 224)))
        img_preprocessed = vgg16.preprocess_input(img_np)
        buffered = BytesIO()
        Image.fromarray(img_preprocessed.astype(np.uint8)).save(buffered, format="PNG")
        return buffered.getvalue()

def clahe(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        img_gray = img.convert('L')
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe = clahe_obj.apply(np.array(img_gray))
        buffered = BytesIO()
        Image.fromarray(img_clahe).save(buffered, format="PNG")
        return buffered.getvalue()

def canny_edge(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        img_gray = img.convert('L')
        edges = cv2.Canny(np.array(img_gray), 100, 200)
        buffered = BytesIO()
        Image.fromarray(edges).save(buffered, format="PNG")
        return buffered.getvalue()

def gaussian_blur(img_bytes, kernel_size=5):
    with Image.open(BytesIO(img_bytes)) as img:
        img_blurred = cv2.GaussianBlur(np.array(img), (kernel_size, kernel_size), 0)
        buffered = BytesIO()
        Image.fromarray(img_blurred).save(buffered, format="PNG")
        return buffered.getvalue()

def resnet_preprocessing(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        img_np = np.array(img.resize((224, 224)))
        img_preprocessed = resnet50.preprocess_input(img_np)
        buffered = BytesIO()
        Image.fromarray(img_preprocessed.astype(np.uint8)).save(buffered, format="PNG")
        return buffered.getvalue()

def inception_preprocessing(img_bytes):
    with Image.open(BytesIO(img_bytes)) as img:
        img_np = np.array(img.resize((299, 299)))  # Inception models typically have an input size of 299x299
        img_preprocessed = inception_v3.preprocess_input(img_np)
        buffered = BytesIO()
        Image.fromarray(img_preprocessed.astype(np.uint8)).save(buffered, format="PNG")
        return buffered.getvalue()
def normalize_to_interval(img_bytes, min_val=0, max_val=1):
    """
    Normalize the image to a specific interval [min_val, max_val].
    """
    with Image.open(BytesIO(img_bytes)) as img:
        img_np = np.array(img, dtype=np.float32)
        img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))  # First, normalize to [0, 1]
        img_normalized = img_np * (max_val - min_val) + min_val  # Then, scale and shift to desired interval
        buffered = BytesIO()
        Image.fromarray(np.clip(img_normalized * 255, 0, 255).astype(np.uint8)).save(buffered, format="PNG")#jpg?
        return buffered.getvalue()
def webp_compression(img_bytes, quality=90):
    """
    Compress the image using the WebP format at a given quality level and then read it back.
    """
    with Image.open(BytesIO(img_bytes)) as img:
        buffered = BytesIO()
        img.save(buffered, format="WEBP", quality=quality)
        # Read the WebP bytes back to an image and return bytes
        with Image.open(buffered) as img_webp:
            buffered_out = BytesIO()
            img_webp.save(buffered_out, format="PNG")
            return buffered_out.getvalue()
def randomized_smoothing(img_bytes, sigma=0.5, num_samples=50):
    """
    Apply randomized smoothing by adding Gaussian noise to the image multiple times
    and then averaging the noisy images.
    """
    with Image.open(BytesIO(img_bytes)) as img:
        img_np = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
        height, width, channels = img_np.shape

        # Accumulator for all noisy images
        accumulator = np.zeros_like(img_np)

        for _ in range(num_samples):
            noise = np.random.normal(0, sigma, (height, width, channels))
            noisy_img = np.clip(img_np + noise, 0, 1)
            accumulator += noisy_img

        # Average the noisy images
        averaged_img = (accumulator / num_samples) * 255.0  # Scale back to [0,255]
        buffered = BytesIO()
        Image.fromarray(np.clip(averaged_img, 0, 255).astype(np.uint8)).save(buffered, format="PNG")
        return buffered.getvalue()

# for width in range(120, 448, 4):  # Example: iterating from 180 to 250 with step size 10
#     for height in range(120, 448, 4):
#         for interp in ['bilinear', 'nearest', 'bicubic']:
#             preprocessors.append((lambda img, w=width, h=height, i=interp: resize_image(img, (w, h), i),
#                                   f"Resize to ({width}, {height}) with {interp} interpolation"))

# Add rotation variations
# for angle in range(1, 361):  # Iterating from 1 to 360
#     preprocessors.append((lambda img, a=angle: rotate_image(img, a), f"Rotate by {angle} degrees"))

# # Add brightness adjustment variations
# for factor in range(50, 151,
#                     1):  # Example: iterating from 0.5 to 1.5 with step size 0.1 (multiplied by 100 for integer loop)
#     factor /= 100
#     preprocessors.append((lambda img, f=factor: adjust_brightness(img, f), f"Adjust brightness by factor {factor}"))
# Initialize a file pointer to the beginning of target_probs.txt.

# Extend preprocessors with the new functions

# JPEG Compression
# for quality in range(0, 101, 1):
#     preprocessors.append((lambda img, q=quality: compress_jpeg(img, q), f"JPEG compression with quality {quality}"))
#
# # PNG Compression
# for level in range(10):
#     preprocessors.append((lambda img, l=level: compress_png(img, l), f"PNG compression with level {level}"))
#
# # GIF Compression (no parameters to vary for basic GIF compression)
# preprocessors.append((compress_gif, "GIF compression"))
#
# # Reduce Color Depth
# preprocessors.append((reduce_color_depth, "Reduce color depth to palette"))
#
# # Cropping
# for left in range(0, 50, 10):
#     for upper in range(0, 50, 10):
#         for right in range(174, 224, 10):
#             for lower in range(174, 224, 10):
#                 preprocessors.append((lambda img, l=left, u=upper, r=right, lo=lower: crop_image(img, l, u, r, lo), f"Crop image to box ({left}, {upper}, {right}, {lower})"))


# Extend preprocessors with the new functions

# Gaussian Blur
# for radius in range(1, 6):
#     preprocessors.append((lambda img, r=radius: gaussian_blur(img, r), f"Gaussian Blur with radius {radius}"))
#
# # Median Blur
# for size in [3, 5, 7]:
#     preprocessors.append((lambda img, s=size: median_blur(img, s), f"Median Blur with size {size}"))
#
# # Contrast Adjustment
# for factor in range(50, 151, 1):
#     factor /= 100
#     preprocessors.append((lambda img, f=factor: adjust_contrast(img, f), f"Contrast adjustment by factor {factor}"))
#
# # Sharpness Adjustment
# for factor in range(50, 151, 1):
#     factor /= 100
#     preprocessors.append((lambda img, f=factor: adjust_sharpness(img, f), f"Sharpness adjustment by factor {factor}"))
#
# # Grayscale Conversion
# preprocessors.append((convert_grayscale, "Convert to Grayscale"))
#
# # Sepia Filter
# preprocessors.append((apply_sepia, "Apply Sepia Filter"))

# Extend preprocessors with the new functions

# Binary Thresholding
# for threshold in range(50, 206, 1):  # Step of 5 from 50 to 205
#     preprocessors.append((lambda img, t=threshold: binary_threshold(img, t), f"Binary Thresholding at {threshold}"))

# Convert to HSV
# preprocessors.append((convert_to_hsv, "Convert to HSV color space"))

# Convert to LAB
# preprocessors.append((convert_to_lab, "Convert to LAB color space"))

# Skewing
# for degree in range(-180, 180, 1):  # Skewing from -25 to 25 degrees
#     preprocessors.append((lambda img, d=degree: skew_image(img, 'x', d), f"Skew in X direction by {degree} degrees"))
#     preprocessors.append((lambda img, d=degree: skew_image(img, 'y', d), f"Skew in Y direction by {degree} degrees"))

# Min-Max Scaling
# preprocessors.append((min_max_scaling, "Min-Max Scaling of pixel values"))

# Extend preprocessors with the new functions

# Channel Splitting
# for channel in ['R', 'G', 'B']:
#     preprocessors.append((lambda img, c=channel: split_channel(img, c), f"Split {channel} channel"))
#
# # Channel Swapping
# for swap in ['RG', 'RB', 'GB']:
#     preprocessors.append((lambda img, s=swap: swap_channels(img, s), f"Swap channels {swap}"))
#
# # Salt and Pepper Noise
# for prob in [i/100.0 for i in range(101)]:
#     preprocessors.append((lambda img, p=prob: add_salt_pepper_noise(img, p), f"Add salt and pepper noise with probability {prob}"))
#
# # Gaussian Noise
# for sigma in [10, 25, 50]:
#     preprocessors.append((lambda img, s=sigma: add_gaussian_noise(img, 0, s), f"Add Gaussian noise with sigma {sigma}"))
#
# # Histogram Equalization
# preprocessors.append((histogram_equalization, "Histogram Equalization"))
#
# # Unsharp Mask
# for radius in [1, 2, 3]:
#     for percent in [100, 150, 200]:
#         preprocessors.append((lambda img, r=radius, p=percent: unsharp_mask(img, r, p), f"Unsharp mask with radius {radius} and percent {percent}"))

# Extend preprocessors with the new functions

# Flip Operations
# preprocessors.append((horizontal_flip, "Horizontal Flip"))
# preprocessors.append((vertical_flip, "Vertical Flip"))
#
# # Zooming
# for factor in [(i+10)/100.0 for i in range(201)]:
#     preprocessors.append((lambda img, f=factor: zoom_image(img, f), f"Zoom by factor {factor}"))
#
# # Gamma Correction
# for gamma in [(i+10)/100.0 for i in range(201)]:
#     preprocessors.append((lambda img, g=gamma: gamma_correction(img, g), f"Gamma correction with gamma {gamma}"))
#
# # Negative Image
# preprocessors.append((negative_image, "Negative Image"))
#
# # Posterization
# for bits in [2, 4, 6]:
#     preprocessors.append((lambda img, b=bits: posterize(img, b), f"Posterize with {bits} bits"))
#
# # Convert to HSV (Corrected)
# preprocessors.append((convert_to_hsv_corrected, "Convert to HSV color space (Corrected)"))
#
# # Convert to LAB (Corrected)
# preprocessors.append((convert_to_lab_corrected, "Convert to LAB color space (Corrected)"))

# for plane in range(8):
#     preprocessors.append((lambda img, p=plane: bit_plane_slicing(img, p), f"Bit-plane slicing at plane {plane}"))

# for gamma in [0.5, 1.5, 3.0]:
#     preprocessors.append((lambda img, g=gamma: gamma_transform(img, g), f"Gamma transform with gamma {gamma}"))

# preprocessors.append((log_transform, "Log Transform"))

# for kernel_size in [3, 5, 7]:
#     preprocessors.append((lambda img, k=kernel_size: morphological_open(img, k), f"Morphological Opening with kernel size {kernel_size}"))
#     preprocessors.append((lambda img, k=kernel_size: morphological_close(img, k), f"Morphological Closing with kernel size {kernel_size}"))

# for _ in range(1):  # Since Otsu doesn't have any variable parameters
#     preprocessors.append((otsu_binarization, "Otsu's Binarization"))

# preprocessors.append((vgg_preprocessing, "VGG Preprocessing"))
#
# preprocessors.append((resnet_preprocessing, "ResNet Preprocessing"))
#
# preprocessors.append((inception_preprocessing, "Inception Preprocessing"))
#
# preprocessors.append((clahe, "CLAHE"))
#
# preprocessors.append((canny_edge, "Canny Edge Detection"))
#
#
# for k in [3, 5, 7]:  # Different kernel sizes
#     preprocessors.append((lambda img, k=k: gaussian_blur(img, k), f"Gaussian Blur with kernel {k}x{k}"))

# Add preprocessors for different normalization intervals using loops
# step = 0.10
# intervals = [(i, j) for i in np.arange(-1, 1+step, step) for j in np.arange(i+step, 1+step, step)]
# for min_val, max_val in intervals:
#     preprocessors.append((lambda img, minv=min_val  , maxv=max_val: normalize_to_interval(img, minv, maxv), f"Normalization to interval [{min_val}, {max_val}]"))

# Add preprocessors for different WebP compression quality levels using a loop
# for quality in range(0, 101, 1):  # From 10% to 100% quality
#     preprocessors.append((lambda img, q=quality: webp_compression(img, q), f"WebP compression with quality {quality}%"))


# Add preprocessors with different sigma values for the Gaussian noise
# for sigma in [0.1, 0.5, 1.0, 2.0]:
#     preprocessors.append((lambda img, s=sigma: randomized_smoothing(img, s), f"Randomized smoothing with sigma {sigma}"))
# file_pointer = open("target_probs.txt", "r")


def query(input_data):
    """
    Query function to read predictions from target_probs.txt and return as a Python object.
    """
    with open("target_probs.txt", "r") as f:
        content = f.read()
        result = ast.literal_eval(content)
    return result



def mobilenet_query(image_bytes, val):
    """
    Query the MobileNetV2 model with the given image bytes.
    Returns the predictions as a list of (score, label) tuples.
    """
    image_np = np.array(Image.open(BytesIO(image_bytes)))

    # Resize the image to 224x224
    img_tensor = tf.convert_to_tensor(image_np)
    img_resized = tf.image.resize(img_tensor, [224, 224], method=tf.image.ResizeMethod.BICUBIC)

    preprocessed_img = (img_resized - val) / val

    predictions = model(tf.expand_dims(preprocessed_img, axis=0))
    sorted_predictions = sorted(enumerate(predictions.numpy()[0]), key=lambda x: x[1], reverse=True)

    return [(score, f'class_{label}') for label, score in sorted_predictions]

def compare_outputs(input_data, labels):
    """
    Compare the outputs of MobileNetV2 and the target_probs.txt for the given image bytes and provide verbose output.
    """
    original_response_mobilenet = mobilenet_query(input_data, 127.5)
    print(original_response_mobilenet)
    mobilenet_label_score_dict = {labels[item[1].split("_")[1]][0]: item[0] for item in original_response_mobilenet}

    for preprocessor, description in preprocessors:
        modified_data = preprocessor(input_data)
        # for nmf in range (0,255,1):
        mobilenet_response = mobilenet_query(modified_data,127.5)
        modified_mobilenet_label_score_dict = {labels[item[1].split("_")[1]][0]: item[0] for item in mobilenet_response}
        api_response = query(base64.b64encode(modified_data))
        api_label_score_dict = {str(item[1]): item[0] for item in api_response['output']}
        diffs = []
        if abs(mobilenet_response[0][0] - api_response['output'][0][0]) > 0.70 or mobilenet_response[0][1] != "class_269" or mobilenet_response[1][1] != "class_271":
            continue

        print(f"Results for {description}:")
        # print("Detailed Differences between MobileNetV2 and target_probs.txt:")

        for label_name, score in mobilenet_label_score_dict.items():
            if label_name in api_label_score_dict:
                diff = abs(score - api_label_score_dict[label_name])
                diffs.append(diff ** 2)
                # if diff > 0:  # Only print labels where there's a difference
                #     print(
                #         f"Label: {label_name}, MobileNetV2 Score: {score}, target_probs.txt Score: {api_label_score_dict[label_name]}, Difference: {diff}")

        diff_sum = sum(diffs)
        print(f"\nSum of Differences (squared emphasis): {diff_sum}")

        # Displaying top 3 scores from MobileNetV2 and API side by side
        print("\nTop 3 Scores Comparison:")
        for mb, api in zip(mobilenet_response[:3], api_response['output'][:3]):
            mb_label_name = labels[mb[1].split('_')[1]][0]
            print(
                f"MobileNetV2: Score {mb[0]:.4f}, Label {mb_label_name} | target_probs.txt: Score {api[0]:.4f}, Label {api[1]}")

        print("\n" + "=" * 50 + "\n")


# Sample usage:
with open('timber_wolf.png', 'rb') as f:
    input_data = f.read()
compare_outputs(input_data, labels)

#rotate 66,67,93,137,139,282,283
#saltpepper 12,16
#zoom 37

#jpeg 0-3%
# Results for Normalization to interval [-0.8, 0.8999999999999996]:
# MobileNetV2: Score 0.2556, Label timber wolf | target_probs.txt: Score 0.2858, Label timber wolf
# Results for Normalization to interval [-0.7000000000000001, 0.5999999999999996]:
# MobileNetV2: Score 0.2569, Label timber wolf | target_probs.txt: Score 0.2858, Label timber wolf
# Results for Normalization to interval [-0.7000000000000001, 0.6999999999999997]:
# MobileNetV2: Score 0.3029, Label timber wolf | target_probs.txt: Score 0.2858, Label timber wolf
# Results for Normalization to interval [-0.7000000000000001, 0.7999999999999996]:
# MobileNetV2: Score 0.3301, Label timber wolf | target_probs.txt: Score 0.2858, Label timber wolf
# Results for Normalization to interval [-0.6000000000000001, 0.49999999999999967]:
# MobileNetV2: Score 0.3030, Label timber wolf | target_probs.txt: Score 0.2858, Label timber wolf
# Results for Normalization to interval [-0.5000000000000001, 0.2999999999999997]:
# MobileNetV2: Score 0.2490, Label timber wolf | target_probs.txt: Score 0.2858, Label timber wolf
# Results for Normalization to interval [-0.40000000000000013, 0.19999999999999984]:
# MobileNetV2: Score 0.2571, Label timber wolf | target_probs.txt: Score 0.2858, Label timber wolf
# Results for Normalization to interval [0.49999999999999956, 0.5999999999999995]:
# MobileNetV2: Score 0.3279, Label timber wolf | target_probs.txt: Score 0.2858, Label timber wolf


