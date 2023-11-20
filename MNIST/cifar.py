import time
import requests
from itertools import permutations, product
import numpy as np
from keras.datasets import cifar10, cifar100
from collections import Counter

from scipy.stats import mode


# # 1. Data Loading Module
def load_data(choice):
    if choice == 0:  # CIFAR-100: first 10,000 combined
        print("First 10000 train dataset100")
        (x_train, y_train), _ = cifar100.load_data()

        print(len(x_train))
        print(len(y_train))
        return x_train[:10000], y_train[:10000]
    elif choice == 1:  # CIFAR-100: train only
        print("Train dataset100")
        (x_train, y_train), _ = cifar100.load_data()
        print(len(x_train))
        print(len(y_train))
        return x_train, y_train
    elif choice == 2:  # CIFAR-100: test only
        print("Test dataset100")
        _, (x_test, y_test) = cifar100.load_data()
        print(len(x_test))
        print(len(y_test))
        return x_test, y_test
    elif choice == 3:  # CIFAR-100: both combined
        print("Combined dataset100")
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_combined = np.concatenate((x_train, x_test), axis=0)
        y_combined = np.concatenate((y_train, y_test), axis=0)
        print(len(x_combined))
        print(len(y_combined))
        return x_combined, y_combined
    elif choice == 10:  # CIFAR-10: first 10,000 combined
        print("First 10000 train dataset10")
        (x_train, y_train), _ = cifar10.load_data()

        print(len(x_train))
        print(len(y_train))
        return x_train[:10000], y_train[:10000]
    elif choice == 20:  # CIFAR-10: train only
        print("Train dataset10")
        (x_train, y_train), _ = cifar10.load_data()
        print(len(x_train))
        print(len(y_train))
        return x_train, y_train
    elif choice == 30:  # CIFAR-10: test only
        print("Test dataset10")
        _, (x_test, y_test) = cifar10.load_data()
        print(len(x_test))
        print(len(y_test))
        return x_test, y_test
    elif choice == 40:  # CIFAR-10: both combined
        print("Combined dataset10")
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_combined = np.concatenate((x_train, x_test), axis=0)
        y_combined = np.concatenate((y_train, y_test), axis=0)
        print(len(x_combined))
        print(len(y_combined))
        return x_combined, y_combined
    elif choice == 4:  # CIFAR-100: first 100 train
        (x_train, y_train), _ = cifar100.load_data()
        return x_train[:100], y_train[:100]
    elif choice == 5:  # CIFAR-100: first 100 test
        _, (x_test, y_test) = cifar100.load_data()
        return x_test[:100], y_test[:100]
    elif choice == 6:  # CIFAR-10: first 100 train
        (x_train, y_train), _ = cifar10.load_data()
        return x_train[:100], y_train[:100]
    elif choice == 7:  # CIFAR-10: first 100 test
        _, (x_test, y_test) = cifar10.load_data()
        return x_test[:100], y_test[:100]
    else:
        raise ValueError("Invalid choice. Choose from 0, 1, 2, 3, 4, or 5.")


# 3. Class Position Module
def add_class_position(data, start_from_zero=True):
    # Extracting the shape to ensure we handle the 3D data
    num_classes, num_channels, num_combinations = data.shape

    if start_from_zero:
        positions = np.arange(num_classes).reshape(-1, 1)
    else:
        positions = np.arange(1, num_classes + 1).reshape(-1, 1)

    # Repeat the positions for each combination
    positions_repeated = np.repeat(positions, num_combinations, axis=1).reshape(num_classes, 1, num_combinations)

    # Stack along the second dimension (channels) but now at the beginning
    return np.hstack([positions_repeated, data])


def reverse_rows(matrices):
    """Return the reversed version of the input matrices."""
    return [np.flip(m, axis=0) for m in matrices]


# Function to permute the columns of the matrix
def shift_columns(matrix):
    num_columns = matrix.shape[1]
    all_matrices = [matrix]

    # Shift the matrix columns to the right 3 times
    for i in range(1, 4):
        shifted_matrix = np.roll(matrix, shift=i, axis=1)
        all_matrices.append(shifted_matrix)

    # Reverse the rows for each of the 4 matrices
    reversed_matrices = reverse_rows(all_matrices)

    return all_matrices + reversed_matrices


def query(input_data):
    # Convert numpy ndarray to a list
    input_list = input_data.tolist()

    response = requests.post('http://count-cifar.advml.com/score', json={'data': input_list})
    return response.json()


# 2. Metric Computation Module
# counts how many RGB have combinations of exactly 125,245,0
def compute_rgb_metric_combinations(data, labels):
    combinations = list(product([125, 245, 0], repeat=3))
    results = np.zeros((100, 3, len(combinations)), dtype=int)

    for idx, (r_val, g_val, b_val) in enumerate(combinations):
        for i in range(100):
            class_images = data[labels[:, 0] == i]
            r_count = np.sum(class_images[:, :, :, 0] == r_val)
            g_count = np.sum(class_images[:, :, :, 1] == g_val)
            b_count = np.sum(class_images[:, :, :, 2] == b_val)
            results[i, :, idx] = [r_count, g_count, b_count]

    return results


def compute_rgb_metric_combinations(data, labels):
    # Define the combinations of ranges
    combinations = [
        (125, 245),
        (0, 255),
        (125, 244),
        (124, 244),
        (124, 245),
        (1, 255),
    ]

    # Storage for results
    results_sum = np.zeros((100, 3, len(combinations)), dtype=int)
    results_unique = np.zeros((100, 3, len(combinations)), dtype=int)
    results_unique_total = np.zeros((100, 3, len(combinations)), dtype=int)

    # Unique metric computation (existing)
    for idx, (min_val, max_val) in enumerate(combinations):
        for i in range(100):
            # print(f"{idx} {i} one")
            class_images = data[labels[:, 0] == i]
            unique_counts = [set(), set(), set()]  # For R, G, and B channels
            for image in class_images:
                for channel in range(3):
                    unique_pixels = set(image[:, :, channel].ravel())
                    in_range_pixels = {pixel for pixel in unique_pixels if min_val <= pixel <= max_val}
                    unique_counts[channel].update(in_range_pixels)
            results_unique[i, :, idx] = [len(unique_counts[0]), len(unique_counts[1]), len(unique_counts[2])]

    # Unique total metric computation (new)
    for idx, (min_val, max_val) in enumerate(combinations):
        for i in range(100):
            # print(f"{idx} {i} two")
            class_images = data[labels[:, 0] == i]
            unique_counts = [0, 0, 0]  # For R, G, and B channels
            seen_pixels = [set(), set(), set()]  # For R, G, and B channels to avoid double counting
            for image in class_images:
                for channel in range(3):
                    unique_pixels = set(image[:, :, channel].ravel())
                    in_range_pixels = {pixel for pixel in unique_pixels if min_val <= pixel <= max_val}
                    new_pixels = in_range_pixels - seen_pixels[channel]
                    unique_counts[channel] += len(new_pixels)
                    seen_pixels[channel].update(new_pixels)
            results_unique_total[i, :, idx] = unique_counts

    # Combining all metrics
    results = np.concatenate([results_sum, results_unique, results_unique_total], axis=2)
    return results


# 614400 number
# def compute_rgb_metric_combinations(data, labels):
#     # Define the intervals inside the function
#     combinations = [
#         (125,245),
#         (125,246),
#         (0,125),
#         (0,124),
#         (245,255),
#         (246,255),
#     ]
#
#     results = np.zeros((100, 3, len(combinations)), dtype=int)
#
#     for idx, interval in enumerate(combinations):
#         print(interval)
#         for i in range(100):
#             class_images = data[labels[:, 0] == i]
#
#             r_count = np.sum((class_images[:, :, :, 0] >= interval[0]) & (class_images[:, :, :, 0] <= interval[1]))
#             g_count = np.sum((class_images[:, :, :, 1] >= interval[0]) & (class_images[:, :, :, 1] <= interval[1]))
#             b_count = np.sum((class_images[:, :, :, 2] >= interval[0]) & (class_images[:, :, :, 2] <= interval[1]))
#
#             results[i, :, idx] = [ r_count, g_count, b_count]
#
#     return results

def compute_rgb_metric_combinations(data, labels, thresholds=[125, 245, 0]):
    # Initialize a matrix to store counts for each category
    counts_matrix = np.zeros((100, 3), dtype=int)

    for idx, unique_label in enumerate(np.unique(labels)):
        # Filter images that belong to the current label
        category_images = [data[i] for i, label in enumerate(labels) if label == unique_label]

        # Aggregate all R, G, B values above their respective thresholds
        red_values = [pixel[0] for image in category_images for row in image for pixel in row if
                      pixel[0] > thresholds[0]]
        green_values = [pixel[1] for image in category_images for row in image for pixel in row if
                        pixel[1] > thresholds[1]]
        blue_values = [pixel[2] for image in category_images for row in image for pixel in row if
                       pixel[2] > thresholds[2]]

        # Store the counts in the matrix
        counts_matrix[idx] = [len(red_values), len(green_values), len(blue_values)]

    return counts_matrix


def most_common_pixels_cifar10(data):
    # Flatten the dataset
    flattened_data = data.reshape(-1, 3)
    # Find unique rows and their counts
    unique_rows, counts = np.unique(flattened_data, axis=0, return_counts=True)
    # Sort by counts and take the top 100
    sorted_indices = np.argsort(-counts)
    common_pixels = unique_rows[sorted_indices[:100]]
    return common_pixels


def most_common_pixel_per_class_cifar100(data, labels):
    common_pixels = []
    flattened_labels = labels.flatten()
    for class_id in range(100):
        # Filter the images belonging to the current class
        class_data = data[flattened_labels == class_id]
        # Flatten the dataset for the class
        flattened_data = class_data.reshape(-1, 3)
        # Find unique rows and their counts
        unique_rows, counts = np.unique(flattened_data, axis=0, return_counts=True)
        # Sort by counts and take the top pixel
        top_pixel = unique_rows[np.argmax(counts)]
        common_pixels.append(top_pixel)
    return np.array(common_pixels)


# def compute_rgb_metric_combinations():
#     # Load the datasets
#     (x_train_10, _), (x_test_10, _) = cifar10.load_data()
#     (x_train_100, y_train_100), (x_test_100, y_test_100) = cifar100.load_data()
#
#     # For CIFAR-10
#     train_common_10 = most_common_pixels_cifar10(x_train_10)
#     test_common_10 = most_common_pixels_cifar10(x_test_10)
#     combined_common_10 = most_common_pixels_cifar10(np.vstack([x_train_10, x_test_10]))
#
#     # For CIFAR-100
#     train_common_100 = most_common_pixel_per_class_cifar100(x_train_100, y_train_100)
#     test_common_100 = most_common_pixel_per_class_cifar100(x_test_100, y_test_100)
#     combined_common_100 = most_common_pixel_per_class_cifar100(np.vstack([x_train_100, x_test_100]),
#                                                               np.vstack([y_train_100, y_test_100]))
#     rgb_data_combinations = np.stack([train_common_10, test_common_10, combined_common_10,
#                                       train_common_100, test_common_100, combined_common_100], axis=2)
#
#     res = rgb_data_combinations
#     print(res)
#     # Return the list of 100x3 matrices
#     return res
def compute_rgb_dominance(data, labels):
    # Initialize a matrix to store counts for each category
    # Assuming there are as many categories as there are unique labels
    num_categories = len(np.unique(labels))
    counts_matrix = np.zeros((num_categories, 3), dtype=int)

    # Iterate over each unique label
    for idx, unique_label in enumerate(np.unique(labels)):
        # Filter images that belong to the current label
        category_images = [data[i] for i, label in enumerate(labels) if label == unique_label]

        # Count pixels where the red channel is dominant
        red_dominant_count = sum(
            1 for image in category_images for row in image for pixel in row if pixel[0] > 125 and pixel[0] > 125
        )

        # The question asks only for red, but if we want to count for green and blue as well:
        green_dominant_count = sum(
            1 for image in category_images for row in image for pixel in row if pixel[1] > 245 and pixel[1] > 245
        )

        blue_dominant_count = sum(
            1 for image in category_images for row in image for pixel in row if pixel[2] > 0 and pixel[2] > 0
        )

        # Store the counts in the matrix
        counts_matrix[idx] = [red_dominant_count, green_dominant_count, blue_dominant_count]

    return counts_matrix


def count_most_common_pixels(images, labels):
    # Flatten the labels if they are not already one-dimensional
    labels = labels.flatten()

    # Initialize a list to store the most common pixel for each category
    num_categories = 100  # CIFAR-100 has 100 fine label classes
    most_common_pixels = np.zeros((num_categories, 3), dtype=int)

    # Iterate over each category
    for i in range(num_categories):
        # Select images of the current class using boolean indexing
        class_images = images[labels == i]

        # Reshape the images to a 2D array where each row is a pixel
        pixels = class_images.reshape(-1, 3)

        # Find the most common pixel
        most_common_pixel = mode(pixels, axis=0).mode

        # Store the most common pixel in the list
        most_common_pixels[i] = most_common_pixel

    return most_common_pixels


def count_dominant_hue_images_per_class(images, labels):
    # Ensure labels are flattened to a 1D array for processing
    labels = labels.flatten()

    # Initialize a matrix to store counts for each category
    num_categories = 100  # CIFAR-100 has 100 fine label classes
    dominant_hue_counts = np.zeros((num_categories, 3), dtype=int)

    # Iterate over each category
    for i in range(num_categories):
        # Select images of the current class using boolean indexing
        class_images = images[labels == i]

        # Sum the color intensities for each image and determine the dominant hue
        color_sums = class_images.sum(axis=(1, 2))  # Sum over height and width, retain the color channel

        # Count the number of images where each color channel is the highest sum
        dominant_hue_counts[i, 0] = np.sum(color_sums[:, 0] >= np.maximum(color_sums[:, 1], color_sums[:, 2]))  # Red is dominant
        dominant_hue_counts[i, 1] = np.sum(color_sums[:, 1] >= np.maximum(color_sums[:, 0], color_sums[:, 2]))  # Green is dominant
        dominant_hue_counts[i, 2] = np.sum(color_sums[:, 2] >= np.maximum(color_sums[:, 0], color_sums[:, 1]))  # Blue is dominant

    return dominant_hue_counts

# Assuming x_data and y_data are available, call the function like this:
# dominant_hue_image_counts = count_dominant_hue_images_per_class(x_data, y_data)


# [[[75101]
#   [20784]
#   [13039]]
def count_majority_color_images_per_class(images, labels, threshold=0.5):
    # Ensure labels are flattened to a 1D array for processing
    labels = labels.flatten()

    # Initialize a matrix to store counts for each category
    num_categories = 100  # CIFAR-100 has 100 fine label classes
    majority_color_counts = np.zeros((num_categories, 3), dtype=int)

    # Total number of pixels per image
    total_pixels = images.shape[1] * images.shape[2]

    # Iterate over each category
    for i in range(num_categories):
        # Select images of the current class using boolean indexing
        class_images = images[labels == i]

        # Count the number of images with more than threshold% of red, green, or blue pixels
        red_channel = class_images[:,:,:,0]
        green_channel = class_images[:,:,:,1]
        blue_channel = class_images[:,:,:,2]

        majority_color_counts[i, 0] = np.sum((red_channel > 128).reshape(class_images.shape[0], -1).sum(axis=1) > threshold * total_pixels)
        majority_color_counts[i, 1] = np.sum((green_channel > 128).reshape(class_images.shape[0], -1).sum(axis=1) > threshold * total_pixels)
        majority_color_counts[i, 2] = np.sum((blue_channel > 128).reshape(class_images.shape[0], -1).sum(axis=1) > threshold * total_pixels)

    return majority_color_counts
def count_top_left_pixel_specific_values(images, labels):
    # Ensure labels are flattened to a 1D array for processing
    labels = labels.flatten()

    # Initialize a matrix to store counts for each category
    num_categories = 100  # CIFAR-100 has 100 fine label classes
    top_left_pixel_counts = np.zeros((num_categories, 3), dtype=int)

    # Check the top-left pixel (position [0, 0] in each image)
    top_left_pixels = images[:, 0, 0, :]

    # Iterate over each category
    for i in range(num_categories):
        # Select the top-left pixels of the current class using boolean indexing
        class_top_left_pixels = top_left_pixels[labels == i]

        # Count the number of images with the top-left pixel having specific values
        top_left_pixel_counts[i, 0] = np.sum(class_top_left_pixels[:, 0] == 125)  # Red channel is 125
        top_left_pixel_counts[i, 1] = np.sum(class_top_left_pixels[:, 1] == 245)  # Green channel is 245
        top_left_pixel_counts[i, 2] = np.sum(class_top_left_pixels[:, 2] == 0)    # Blue channel is 0

    return top_left_pixel_counts
def count_specific_rgb_occurrences_per_class(images, labels):
    # Ensure labels are flattened to a 1D array for processing
    labels = labels.flatten()

    # Initialize a matrix to store counts for each category
    num_categories = 100  # CIFAR-100 has 100 fine label classes
    rgb_occurrence_counts = np.zeros((num_categories, 1), dtype=int)

    # Define the RGB value to look for
    target_rgb = np.array([125, 245, 0])

    # Iterate over each category
    for i in range(num_categories):
        # Select images of the current class using boolean indexing
        class_images = images[labels == i]

        # Flatten the images to a 2D array where each row is a pixel
        class_pixels = class_images.reshape(-1, 3)

        # Count the occurrences of the target RGB value
        rgb_occurrence_counts[i] = np.sum(np.all(class_pixels == target_rgb, axis=1))

    return rgb_occurrence_counts
# 4. Main Orchestrator
def count_color_channel_distribution(images, labels):
    # Flatten the labels to ensure it is one-dimensional
    labels = labels.flatten()

    # Initialize the result array
    result = np.zeros((100, 3))  # 100 classes and 3 color channels

    # Iterate through each class
    for i in range(100):
        class_images = images[labels == i]  # Get all images of class i

        # Calculate the mean of each channel
        red_mean = class_images[:, :, :, 0].mean()
        green_mean = class_images[:, :, :, 1].mean()
        blue_mean = class_images[:, :, :, 2].mean()

        # Determine which range each mean falls into and increment the respective count
        result[i, 0] += (red_mean >= 125) and (red_mean <= 245)
        result[i, 1] += (green_mean >= 125) and (green_mean <= 245)
        result[i, 2] += (blue_mean >= 125) and (blue_mean <= 245)

    return result
def count_pixels_with_specific_combinations(images, labels):
    # Flatten the labels to ensure it is one-dimensional
    labels = labels.flatten()

    # Initialize a matrix to store counts for each category
    num_categories = 100  # CIFAR-100 has 100 fine label classes
    counts_matrix = np.zeros((num_categories, 3), dtype=int)

    # Define the specific values to check for
    red_specific = (245, 0)  # G=245, B=0
    green_specific = (125, 0)  # R=125, B=0
    blue_specific = (125, 245)  # R=125, G=245

    # Iterate over each category
    for i in range(num_categories):
        # Select images of the current class using boolean indexing
        class_images = images[labels == i]

        # Flatten the class images to a list of RGB pixels
        class_pixels = class_images.reshape(-1, 3)

        # Count the occurrences of the specific combination for each channel
        counts_matrix[i, 0] = np.sum((class_pixels[:, 1] == red_specific[0]) & (class_pixels[:, 2] == red_specific[1]))  # Count for Red
        counts_matrix[i, 1] = np.sum((class_pixels[:, 0] == green_specific[0]) & (class_pixels[:, 2] == green_specific[1]))  # Count for Green
        counts_matrix[i, 2] = np.sum((class_pixels[:, 0] == blue_specific[0]) & (class_pixels[:, 1] == blue_specific[1]))  # Count for Blue

    return counts_matrix


def count_pixels_specific_channels(images, labels):
    # Ensure labels are flattened to a 1D array for processing
    labels = labels.flatten()

    # Initialize a matrix to store counts for each category
    num_categories = 100  # CIFAR-100 has 100 fine label classes
    counts_matrix = np.zeros((num_categories, 3), dtype=int)

    # Iterate over each category
    for idx in range(num_categories):
        # Extract images for the current class
        class_images = images[labels == idx]

        # Flatten the images to a 2D array of pixels for that class
        class_pixels = class_images.reshape(-1, 3)

        # Count for Red: Green channel at 245 and Blue channel at 0
        red_count = np.sum((class_pixels[:, 1] == 245) & (class_pixels[:, 2] == 0))

        # Count for Green: Red channel at 125 and Blue channel at 0
        green_count = np.sum((class_pixels[:, 0] == 125) & (class_pixels[:, 2] == 0))

        # Count for Blue: Red channel at 125 and Green channel at 245
        blue_count = np.sum((class_pixels[:, 0] == 125) & (class_pixels[:, 1] == 245))

        # Populate the counts matrix
        counts_matrix[idx, 0] = red_count
        counts_matrix[idx, 1] = green_count
        counts_matrix[idx, 2] = blue_count

    return counts_matrix
def generate_all_matrices_combinations():
    itrx = 0
    #     btrx = 3888
    btrx = 0
    dataset_choices = [0,1,2,3,4,5,6,7]
    position_starts = [False, True]
    all_result_matrices = []

    for choice in dataset_choices:
        print(f"Choice is {choice}")
        x_data, y_data = load_data(choice)
        print(f"xdata {x_data.shape}")
        print(f"ydata {y_data.shape}")
        rgb_data_combinations = count_pixels_specific_channels(x_data, y_data).reshape(100, 3, 1)
        # print(f"Data is {rgb_data_combinations}")

        for start_from_zero in position_starts:

            all_result_matrices = []
            for idx in range(rgb_data_combinations.shape[2]):
                base_matrix = add_class_position(rgb_data_combinations[:, :, idx:idx + 1], start_from_zero)
                base_matrix = base_matrix.squeeze(axis=2)  # Reduce the 3rd dimension after slicing
                permuted_matrices = shift_columns(base_matrix)
                all_result_matrices.extend(permuted_matrices)
            #                 print (f"{choice} : {start_from_zero} : {idx} \n {base_matrix}\n")
            #                 a = 1/0
            for k in all_result_matrices:
                itrx += 1
                if btrx > itrx:
                    continue
                s = 0
                while s != 1:
                    try:
                        res = query(k)
                        if res:
                            print(f"{itrx} : {res}")
                            # print(k)
                            # print("###############################################################")
                            s = 1
                    except:
                        print("sleeping")
                        time.sleep(31)

                # print(k)
                # print('\n\n\n')
    #                 print (all_result_matrices)
    #                 a = 1/0
    return all_result_matrices
#######################################################################################################################
# generate_all_matrices_combinations()




# a = generate_all_matrices_combinations()
def count_rgb_occurrences_interval(x_data, interval):
    # Initialize a list to hold the counts and indices
    rgb_counts_indices = []

    # Define the interval for RGB values
    lower_bound, upper_bound = interval

    # Loop over each image
    for index, image in enumerate(x_data):
        # Count the occurrences of each RGB value within the interval
        r_count = np.sum((image[:, :, 0] >= lower_bound) & (image[:, :, 0] <= upper_bound))
        g_count = np.sum((image[:, :, 1] >= lower_bound) & (image[:, :, 1] <= upper_bound))
        b_count = np.sum((image[:, :, 2] >= lower_bound) & (image[:, :, 2] <= upper_bound))
        # print(index)

        # If any of the RGB values are present in the image within the interval, add the counts and index to the list
        # if r_count > 0 or g_count > 0 or b_count > 0:
        rgb_counts_indices.append([1024-r_count,1024- g_count,1024- b_count, index])

    # Convert to a numpy array
    rgb_counts_indices = np.array(rgb_counts_indices)
    # Sort by the sum of RGB counts (in descending order) and take the top 100
    sums = rgb_counts_indices[:, :3].sum(axis=1)
    sorted_indices = np.argsort(-sums)
    sorted_rgb_counts_indices = rgb_counts_indices[sorted_indices]
    top_100_rgb_counts_indices = sorted_rgb_counts_indices[:100]
    # print(top_100_rgb_counts_indices)

    return top_100_rgb_counts_indices


def count_unique_rgb_values(x_data):
    # Initialize a list to hold the counts of unique values and indices
    unique_rgb_counts_indices = []

    # Loop over each image
    for index, image in enumerate(x_data):
        # Find the number of unique values for each channel
        unique_r = np.unique(image[:, :, 0])
        unique_g = np.unique(image[:, :, 1])
        unique_b = np.unique(image[:, :, 2])

        # Count the number of unique values
        r_count = len(unique_r)
        g_count = len(unique_g)
        b_count = len(unique_b)

        # Append the counts and index to the list
        unique_rgb_counts_indices.append([255-r_count, 255-g_count,255- b_count, index])

    # Convert to a numpy array
    unique_rgb_counts_indices = np.array(unique_rgb_counts_indices)
    # Sort by the sum of unique counts (in descending order) and take the top 100
    sums = unique_rgb_counts_indices[:, :3].sum(axis=1)
    sorted_indices = np.argsort(-sums)
    sorted_unique_rgb_counts_indices = unique_rgb_counts_indices[sorted_indices]
    top_100_unique_rgb_counts_indices = sorted_unique_rgb_counts_indices[:100]

    return top_100_unique_rgb_counts_indices


def sort_by_most_popular_pixel(x_data):
    # Initialize a list to hold the RGB values of the most popular pixel and the image index
    popular_pixel_rgb_index = []

    # Loop over each image
    for index, image in enumerate(x_data):
        # Flatten the image array to 2D (pixels by RGB values)
        pixels = image.reshape(-1, 3)
        # Find the unique pixels and their counts
        unique_pixels, counts = np.unique(pixels, axis=0, return_counts=True)
        # Get the most popular pixel (the one with the highest count)
        most_popular_pixel_index = np.argmax(counts)
        most_popular_pixel = unique_pixels[most_popular_pixel_index]

        # Append the RGB values of the most popular pixel and the image index
        popular_pixel_rgb_index.append(most_popular_pixel.tolist() + [index])

    # Convert to a numpy array for sorting
    popular_pixel_rgb_index = np.array(popular_pixel_rgb_index)
    # Get the counts for sorting
    counts = np.array([np.max(np.unique(image.reshape(-1, 3), axis=0, return_counts=True)[1]) for image in x_data])
    # Sort the array by the counts in descending order
    sorted_indices = np.argsort(-counts)
    # Apply the sorted indices to get the sorted RGB values and indices
    sorted_popular_pixel_rgb_index = popular_pixel_rgb_index[sorted_indices]
    # Take the top 100 entries
    top_100_popular_pixel_rgb_index = sorted_popular_pixel_rgb_index[:100]

    return top_100_popular_pixel_rgb_index

# def count_rgb_occurrences(x_data, rgb_values):
#     # Initialize a list to hold the counts and indices
#     rgb_counts_indices = []
#     # Loop over each image
#     for index, image in enumerate(x_data):
#         # Count the occurrences of each RGB value
#         r_count = np.sum(np.isin(image[:, :, 0], rgb_values))
#         g_count = np.sum(np.isin(image[:, :, 1], rgb_values))
#         b_count = np.sum(np.isin(image[:, :, 2], rgb_values))
#
#         # If any of the RGB values are present in the image, add the counts and index to the list
#         if r_count > 0 or g_count > 0 or b_count > 0:
#             rgb_counts_indices.append([r_count, g_count, b_count, index])
#             # print(r_count, g_count, b_count, index)
#
#     # Convert to a numpy array
#     rgb_counts_indices = np.array(rgb_counts_indices)
#     # Sort by the sum of RGB counts (in descending order) and take the top 100
#     sums = rgb_counts_indices[:, :3].sum(axis=1)
#     sorted_indices = np.argsort(-sums)
#     sorted_rgb_counts_indices = rgb_counts_indices[sorted_indices]
#     top_100_rgb_counts_indices = sorted_rgb_counts_indices[:100]
#     # print(top_100_rgb_counts_indices)
#     return top_100_rgb_counts_indices
#
def compute_rgb_dominance(data):
    # Reshape the data so that each pixel is a row in the 2D array
    reshaped_data = data.reshape(-1, data.shape[-1])

    # Convert RGB pixels to a unique integer assuming 8 bits per channel
    # Use np.int64 to ensure there is no overflow
    flat_pixels = np.int64(reshaped_data[:, 0]) * 256 * 256 + np.int64(reshaped_data[:, 1]) * 256 + np.int64(reshaped_data[:, 2])

    # Use bincount to count the frequency of each unique value
    counts = np.bincount(flat_pixels)

    # Find the top 100 most frequent unique values
    top_100_indices = np.argpartition(counts, -100)[-100:]
    top_100_counts = counts[top_100_indices]

    # Convert the top unique integers back to RGB values
    top_100_pixels = np.column_stack(((top_100_indices // (256 * 256)),
                                      (top_100_indices % (256 * 256) // 256),
                                      (top_100_indices % 256)))

    # Create the final matrix including counts
    top_100_matrix = np.column_stack((top_100_pixels, top_100_counts))

    # Sort the matrix based on the count column in descending order
    top_100_matrix = top_100_matrix[top_100_matrix[:, -1].argsort()[::-1]]

    return top_100_matrix
def count_popular_pixels_in_range(images):
    # Flatten the image array to a list of RGB pixels
    all_pixels = images.reshape(-1, 3)

    # Filter pixels where at least one channel is between the bounds of 125 and 245
    filtered_pixels = all_pixels[
        np.any((all_pixels > 125) & (all_pixels < 245), axis=1)
    ]

    # Find the unique pixels and their counts in the filtered set
    unique_pixels, counts = np.unique(filtered_pixels, axis=0, return_counts=True)

    # Sort the unique pixels by their counts in descending order
    sorted_indices = np.argsort(-counts)
    top_pixels = unique_pixels[sorted_indices][:100]
    top_counts = counts[sorted_indices][:100]

    # Create the output matrix with RGB values and their counts
    top_pixels_counts = np.column_stack((top_pixels, top_counts))

    return top_pixels_counts


def find_top_pixels_with_image_index_reversed(images):
    # Initialize a dictionary to keep track of pixel counts and their first image index
    pixel_counts = {}

    # Iterate through each image
    for index, image in enumerate(images):
        # Iterate through each pixel in the image
        for pixel in image.reshape(-1, 3):
            # Check if the pixel meets the criteria
            if pixel[0] == 125 or pixel[1] == 245 or pixel[2] == 0:
                key = tuple(pixel)
                if key not in pixel_counts:
                    pixel_counts[key] = [0, index]  # Initialize count and image index
                pixel_counts[key][0] += 1  # Increment count

    # Sort the pixel counts by count, descending, and get the top 100
    top_pixels = sorted(pixel_counts.items(), key=lambda item: -item[1][0])[:100]

    # Reverse the sorted list
    top_pixels_reversed = top_pixels[::-1]

    # Create the output matrix
    result_matrix = np.zeros((100, 4), dtype=int)

    # Populate the result matrix with the RGB values and image index
    for i, ((r, g, b), (count, image_index)) in enumerate(top_pixels_reversed):
        result_matrix[i] = [r, g, b, image_index]

    return result_matrix

def count_images_with_color_intensity_range(images, labels):
    # Initialize the results matrix
    results_matrix = np.zeros((100, 4), dtype=int)

    # Iterate over each class
    for class_index in range(100):
        # Filter images for the current class
        class_mask = labels.flatten() == class_index
        class_images = images[class_mask]

        # Initialize counters for images with the highest intensity count within the range for each color
        r_intensity_images = 0
        g_intensity_images = 0
        b_intensity_images = 0

        # Analyze each image
        for image in class_images:
            # Count pixels within the intensity range for R, G, and B
            r_intensity_count = np.sum((image[:, :, 0] >= 125) & (image[:, :, 0] <= 245))
            g_intensity_count = np.sum((image[:, :, 1] >= 125) & (image[:, :, 1] <= 245))
            b_intensity_count = np.sum((image[:, :, 2] >= 125) & (image[:, :, 2] <= 245))

            # Determine which color has the highest count within the intensity range
            if r_intensity_count >= g_intensity_count and r_intensity_count >= b_intensity_count:
                r_intensity_images += 1
            if g_intensity_count >= r_intensity_count and g_intensity_count >= b_intensity_count:
                g_intensity_images += 1
            if b_intensity_count >= r_intensity_count and b_intensity_count >= g_intensity_count:
                b_intensity_images += 1

        # Record the counts for this class
        results_matrix[class_index, 0] = r_intensity_images
        results_matrix[class_index, 1] = g_intensity_images
        results_matrix[class_index, 2] = b_intensity_images

        # Record the class index
        results_matrix[class_index, 3] = class_index

    return results_matrix

# Assuming x_data and y_data are the image and label datasets, you would call the function like this:
# non_zero_rgb_pixel_counts_by_class = count_non_zero_rgb_pixels(x_data, y_data)
def find_top_100_most_popular_pixels(images, low=125, high=245):
    # Flatten the image data to a list of pixels
    pixels = images.reshape(-1, 3)

    # Filter pixels to those where R, G, and B are all within the specified range
    filtered_pixels = pixels[(pixels > low).all(axis=1) & (pixels < high).all(axis=1)]

    # Count the frequency of each pixel
    pixel_counter = Counter(map(tuple, filtered_pixels))

    # Get the top 100 most common pixels and their counts
    top_100_pixels = pixel_counter.most_common(100)

    # Initialize the results array
    results = np.zeros((100, 4), dtype=int)

    for i, (pixel, count) in enumerate(top_100_pixels):
        # Populate the results matrix with the RGB values and the count
        results[i, :3] = pixel
        results[i, 3] = count

    return results


def find_most_popular_pixel_per_class_in_range(images, labels, low=125, high=245):
    num_classes = 100
    # Initialize the results array
    results = np.zeros((num_classes, 4), dtype=int)

    for i in range(num_classes):
        # Extract images for the current class
        class_images = images[labels.flatten() == i]

        # Flatten the image data to a list of pixels
        pixels = class_images.reshape(-1, 3)

        # Filter pixels to those where any of R, G, or B is within the specified range
        filtered_pixels = pixels[
            (pixels[:, 0] == low) |
            (pixels[:, 1] == high) |
            (pixels[:, 2] == 0)
            ]

        # Count the frequency of each pixel
        pixel_counter = Counter(map(tuple, filtered_pixels))

        # Find the most common pixel and its count, if there are any
        if pixel_counter:
            most_common_pixel, count = pixel_counter.most_common(1)[0]
        else:
            # If no pixels in the range, default to (0, 0, 0) with a count of 0
            most_common_pixel, count = (0, 0, 0), 0

        # Populate the results matrix
        results[i, :3] = most_common_pixel
        results[i, 3] = count

    return results


def find_most_popular_pixel_per_class_within_range(images, labels, low=125, high=245):
    num_classes = 100
    # Initialize the results array
    results = np.zeros((num_classes, 4), dtype=int)

    for i in range(num_classes):
        # Extract images for the current class
        class_images = images[labels.flatten() == i]

        # Flatten the image data to a list of pixels
        pixels = class_images.reshape(-1, 3)

        # Filter pixels to those where R, G, and B are within the specified range
        filtered_pixels = pixels[(pixels > low).all(axis=1) & (pixels < high).all(axis=1)]

        # Count the frequency of each pixel
        pixel_counter = Counter(map(tuple, filtered_pixels))

        # Find the most common pixel and its count, if there are any
        if pixel_counter:
            most_common_pixel, count = pixel_counter.most_common(1)[0]
        else:
            # If no pixels in the range, default to (0, 0, 0) with a count of 0
            most_common_pixel, count = (0, 0, 0), 0

        # Populate the results matrix
        results[i, :3] = most_common_pixel
        results[i, 3] = count

    return results


def count_pixels_and_unique_pixels_per_class(images, labels):
    num_classes = 100
    # Initialize the results array
    results = np.zeros((num_classes, 4), dtype=int)

    for i in range(num_classes):
        # Extract images for the current class
        class_images = images[labels.flatten() == i]

        # Initialize counts for R=125, G=245, B=0
        r_count = g_count = b_count = 0

        # Use a set to count unique pixels
        unique_pixels = set()

        for img in class_images:
            # Count pixels with R = 125
            r_count += np.sum(img[:, :, 0] > 125)
            # Count pixels with G = 245
            g_count += np.sum(img[:, :, 1] > 245)
            # Count pixels with B = 0
            b_count += np.sum(img[:, :, 2] > 0)

            # Flatten the image to a 2D array of pixels and add to the unique pixels set
            unique_pixels.update(map(tuple, img.reshape(-1, 3)))

        # Store the counts and the number of unique pixels in the results array
        results[i] = [r_count, g_count, b_count, len(unique_pixels)]

    return results


def count_dominant_and_unique_pixels(images, labels):
    num_classes = 100
    results = np.zeros((num_classes, 4), dtype=int)

    for class_index in range(num_classes):
        # Filter images by class
        class_images = images[labels == class_index]

        # Initialize counts for red, green, and blue dominance
        red_count, green_count, blue_count = 0, 0, 0

        # Initialize a set for unique pixels
        unique_pixels = set()

        for image in class_images:
            # Count red, green, and blue dominant pixels
            red_dominant = np.all(image > np.stack([image[:, :, 1], image[:, :, 2]], axis=-1), axis=2)
            green_dominant = np.all(image > np.stack([image[:, :, 0], image[:, :, 2]], axis=-1), axis=2)
            blue_dominant = np.all(image > np.stack([image[:, :, 0], image[:, :, 1]], axis=-1), axis=2)

            red_count += np.sum(red_dominant)
            green_count += np.sum(green_dominant)
            blue_count += np.sum(blue_dominant)

            # Update unique pixels
            unique_pixels.update(map(tuple, image.reshape(-1, 3)))

        # Store the results
        results[class_index] = [
            red_count,
            green_count,
            blue_count,
            len(unique_pixels)
        ]

    return results

def count_dominant_pixels(images, labels):
    num_classes = 100
    results = np.zeros((num_classes, 4), dtype=int)

    for class_index in range(num_classes):
        # Get the images for the current class
        class_images = images[labels.flatten() == class_index]

        # Initialize counts
        red_dominant_count = 0
        green_dominant_count = 0
        blue_dominant_count = 0
        unique_pixels = set()

        # Count dominant pixels
        for image in class_images:
            # Find dominant pixels for each color
            red_dominant = (image[:, :, 0] >= image[:, :, 1]) & (image[:, :, 0] >= image[:, :, 2])
            green_dominant = (image[:, :, 1] >= image[:, :, 0]) & (image[:, :, 1] >= image[:, :, 2])
            blue_dominant = (image[:, :, 2] >= image[:, :, 0]) & (image[:, :, 2] >= image[:, :, 1])

            # Update counts
            red_dominant_count += red_dominant.sum()
            green_dominant_count += green_dominant.sum()
            blue_dominant_count += blue_dominant.sum()

            # Add unique pixels to set
            unique_pixels.update(map(tuple, image.reshape(-1, 3)))

        # Assign counts to the results array
        results[class_index] = [
            red_dominant_count,
            green_dominant_count,
            blue_dominant_count,
            len(unique_pixels)
        ]

    return results

def count_zero_intensity_pixels(images, labels):
    num_classes = 100
    results = np.zeros((num_classes, 4), dtype=int)

    labels  = labels.flatten()

    for class_index in range(num_classes):
        # Get images for the current class
        class_images = images[labels == class_index]

        # Count pixels with zero intensity for each channel
        r_zero_count = np.sum(class_images[:, :, :, 0] == 0)
        g_zero_count = np.sum(class_images[:, :, :, 1] == 0)
        b_zero_count = np.sum(class_images[:, :, :, 2] == 0)

        # Store the counts and class index in the results array
        results[class_index] = [r_zero_count, g_zero_count, b_zero_count, class_index]

    return results


def top_pixel_per_class(images, labels):
    num_classes = 100
    top_pixels = np.zeros((num_classes, 4), dtype=int)  # Adjusted for 4 columns

    for class_index in range(num_classes):
        # Select images of the class
        class_images = images[labels.flatten() == class_index]

        # Get unique pixels and their counts
        unique_pixels, counts = np.unique(class_images.reshape(-1, 3), axis=0, return_counts=True)

        # Find the most common pixel
        top_pixel_index = np.argmax(counts)
        top_pixel = unique_pixels[top_pixel_index]
        top_pixel_count = counts[top_pixel_index]

        # Store the top pixel RGB values and its count in the class
        top_pixels[class_index] = [*top_pixel, top_pixel_count]

    return top_pixels


def top_pixel_overall(images):
    # Get unique pixels and their counts for all images
    unique_pixels, counts = np.unique(images.reshape(-1, 3), axis=0, return_counts=True)

    # Sort pixels by count
    sorted_indices = np.argsort(-counts)
    sorted_pixels = unique_pixels[sorted_indices][:100]
    sorted_counts = counts[sorted_indices][:100]

    top_pixels_overall = np.zeros((100, 4), dtype=int)  # Adjusted for 4 columns

    # Store the top 100 pixels RGB values and their counts
    for i in range(100):
        top_pixels_overall[i] = [*sorted_pixels[i], sorted_counts[i]]

    return top_pixels_overall


def top_pixel_per_class_presence(images, labels):
    num_classes = 100
    top_pixels = np.zeros((num_classes, 4), dtype=int)  # Adjusted for 4 columns

    for class_index in range(num_classes):
        # Select images of the class
        class_images = images[labels.flatten() == class_index]

        # Get unique pixels and their counts
        unique_pixels, counts = np.unique(class_images.reshape(-1, 3), axis=0, return_counts=True)

        # Find the most common pixel
        top_pixel_index = np.argmax(counts)
        top_pixel = unique_pixels[top_pixel_index]

        # Count how many images contain the top pixel at least once
        contains_top_pixel = np.sum(np.any(np.all(class_images == top_pixel, axis=(2)), axis=(1)))

        # Store the top pixel RGB values and its count in the class
        top_pixels[class_index] = [*top_pixel, contains_top_pixel]

    return top_pixels


def top_pixel_overall_presence(images):
    # Get unique pixels and their counts for all images
    unique_pixels, counts = np.unique(images.reshape(-1, 3), axis=0, return_counts=True)

    # Sort pixels by count
    sorted_indices = np.argsort(-counts)
    sorted_pixels = unique_pixels[sorted_indices][:100]

    top_pixels_overall = np.zeros((100, 4), dtype=int)  # Adjusted for 4 columns

    # Count how many images contain each of the top 100 pixels at least once
    for i, pixel in enumerate(sorted_pixels):
        contains_pixel = np.sum(np.any(np.all(images == pixel, axis=(2)), axis=(1)))
        top_pixels_overall[i] = [*pixel, contains_pixel]

    return top_pixels_overall


def top_pixel_per_class_presence(images, labels):
    num_classes = 100
    top_pixels = np.zeros((num_classes, 4), dtype=int)

    for class_index in range(num_classes):
        # Select images of the class
        class_images = images[labels.flatten() == class_index]

        # Flatten the images to a 2d array and get unique pixels with counts
        pixels, counts = np.unique(class_images.reshape(-1, 3), axis=0, return_counts=True)

        # Find the most common pixel
        most_common_pixel_idx = np.argmax(counts)
        most_common_pixel = pixels[most_common_pixel_idx]

        # Check how many images have the most common pixel at least once
        images_with_pixel = np.any(np.all(class_images == most_common_pixel, axis=-1), axis=(1, 2))
        num_images_with_pixel = np.sum(images_with_pixel)

        # Store the results
        top_pixels[class_index, :3] = most_common_pixel
        top_pixels[class_index, 3] = num_images_with_pixel

    return top_pixels


def top_pixel_overall_presence(images):
    # Flatten the images to a 2d array and get unique pixels with counts
    pixels, counts = np.unique(images.reshape(-1, 3), axis=0, return_counts=True)

    # Sort the pixels by count and take the top 100
    sorted_indices = np.argsort(-counts)
    top_pixels = pixels[sorted_indices][:100]
    top_counts = counts[sorted_indices][:100]

    # Prepare the result array
    result = np.zeros((100, 4), dtype=int)

    # Check how many images have each of these pixels at least once
    for i, pixel in enumerate(top_pixels):
        images_with_pixel = np.any(np.all(images == pixel, axis=-1), axis=(1, 2))
        num_images_with_pixel = np.sum(images_with_pixel)

        # Store the results
        result[i, :3] = pixel
        result[i, 3] = num_images_with_pixel

    return result
def find_popular_pixels_and_counts(images):
    # Flatten the image data to a list of pixels
    flattened_pixels = images.reshape(-1, 3)

    # Count occurrences of each RGB value
    pixel_frequencies = Counter(map(tuple, flattened_pixels))

    # Identify the top 100 most common pixels
    top_100_pixels = [pixel for pixel, _ in pixel_frequencies.most_common(100)]

    # Function to check if an image contains a given pixel
    def contains_pixel(image, pixel):
        return np.any(np.all(image == np.array(pixel), axis=-1))

    # Prepare the result matrix
    result_matrix = np.zeros((100, 4), dtype=int)

    # Count how many images contain each of the top 100 pixels
    for i, pixel in enumerate(top_100_pixels):
        count = sum(contains_pixel(image, pixel) for image in images)
        result_matrix[i, :3] = pixel
        result_matrix[i, 3] = count

    return result_matrix


def count_pixel_ranges_per_class(images, labels):
    # Define the pixel value ranges
    ranges = [(0, 63), (64, 127), (128, 191), (192, 255)]
    # Initialize the result array
    class_range_counts = np.zeros((100, 4), dtype=int)

    for class_index in range(100):
        # Select images of the current class
        class_images = images[labels.flatten() == class_index]
        # Flatten the images to 2D array for pixel operations
        flat_images = class_images.reshape(-1, 3)

        # For each range, count the pixels that fall within the range for each channel
        for range_index, (low, high) in enumerate(ranges):
            # Use a boolean mask to find pixels within the current range
            mask = (flat_images >= low) & (flat_images <= high)
            # Sum over all pixels that fall within the range for any channel
            class_range_counts[class_index, range_index] = np.sum(np.any(mask, axis=1))

    return class_range_counts
def count_intensity_ranges_per_class(images, labels):
    # Define the ranges for intensity values
    intensity_ranges = [(0, 63), (64, 127), (128, 191), (192, 255)]
    # Initialize an array to hold the count of pixels in each range per class
    counts_per_class = np.zeros((100, 4), dtype=int)

    labels = labels.flatten()

    for class_index in range(100):
        # Extract images for the current class
        class_images = images[labels == class_index]
        # Flatten the image data to a 2D array of pixels for easier processing
        pixels = class_images.reshape(-1, 3)

        # Count pixels within each intensity range for any color channel
        for i, (lower, upper) in enumerate(intensity_ranges):
            # Use a mask to find pixels within the current range for any color channel
            mask = (pixels >= lower) & (pixels <= upper)
            # Sum the mask to count pixels that fall within the range, across all channels
            counts_per_class[class_index, i] = np.sum(mask)

        # The fourth column could be the sum of the first three columns or another metric
        counts_per_class[class_index, 3] = np.sum(counts_per_class[class_index, :3])

    return counts_per_class
def count_specific_pixel_values(images, labels):
    # Initialize counts array
    counts = np.zeros((100, 4), dtype=int)
    labels = labels.flatten()
    # Loop over each class
    for class_index in range(100):
        # Extract images for the current class
        class_images = images[labels == class_index]
        # Flatten the image data to a 2D array of pixels for easier processing
        pixels = class_images.reshape(-1, 3)

        # Count pixels at minimum, midpoint, and maximum value for each color channel
        counts[class_index, 0] = np.sum(pixels != 125)
        counts[class_index, 1] = np.sum(pixels != 245)
        counts[class_index, 2] = np.sum(pixels != 0)

        # Count unique colors in the class
        unique_colors = np.unique(pixels, axis=0)
        counts[class_index, 3] = len(unique_colors)

    return counts


def count_most_common_pixels(images, labels):
    # Initialize the counts matrix
    counts = np.zeros((100, 4), dtype=int)

    # Iterate through each class
    for class_idx in range(100):
        # Extract the images for this class
        class_images = images[labels.flatten() == class_idx]

        # Flatten the image data to a 2D array of pixels for easier processing
        pixels = class_images.reshape(-1, 3)

        # Find the most common pixel value for each color channel
        for channel in range(3):
            values, counts_occurrences = np.unique(pixels[:, channel], return_counts=True)
            most_common_value_index = np.argmax(counts_occurrences)
            most_common_value = values[most_common_value_index]
            counts[class_idx, channel] = counts_occurrences[most_common_value_index]

        # The fourth column is the class index
        counts[class_idx, 3] = class_idx

    return counts


def count_color_presence(images, labels):
    # Initialize the counts matrix
    counts = np.zeros((100, 4), dtype=int)

    # Iterate through each class
    for class_idx in range(100):
        # Extract the images for this class
        class_images = images[labels.flatten() == class_idx]

        # Flatten the images to just a list of pixels
        pixels = class_images.reshape(-1, 3)

        # Count the black pixels
        counts[class_idx, 0] = np.sum(np.all(pixels == 0, axis=1))

        # Count the white pixels
        counts[class_idx, 1] = np.sum(np.all(pixels == 255, axis=1))

        # Count the gray pixels (where all RGB values are equal)
        counts[class_idx, 2] = np.sum(
            np.all(pixels[:, 0] == pixels[:, 1], axis=0) & np.all(pixels[:, 1] == pixels[:, 2], axis=0))

        # The fourth column is the class index
        counts[class_idx, 3] = class_idx

    return counts
def count_color_equality_per_class(images, labels):
    # Initialize the counts matrix
    counts = np.zeros((100, 4), dtype=int)

    # Flatten the labels to ensure it is one-dimensional
    labels = labels.flatten()

    # Iterate through each class
    for class_idx in range(100):
        # Extract the images for this class
        class_images = images[labels == class_idx]

        # Flatten the image data to a 2D array of pixels for easier processing
        pixels = class_images.reshape(-1, 3)

        # Count the pixels where B = G
        counts[class_idx, 0] = np.sum(pixels[:, 1] == pixels[:, 2])

        # Count the pixels where R = B
        counts[class_idx, 1] = np.sum(pixels[:, 0] == pixels[:, 2])

        # Count the pixels where R = G
        counts[class_idx, 2] = np.sum(pixels[:, 0] == pixels[:, 1])

        # The fourth column is the class index
        counts[class_idx, 3] = class_idx

    return counts

dataset_choices = [0, 1, 2, 3, 10, 20, 30, 40,4,5,6,7]

for choice in dataset_choices:
    for rgb in [(125, 245, 0)]:
        print(choice)
        x_data, y_data = load_data(choice)
        # print(f"{choice} {len(x_data)} {len(y_data)}")
        rasp = count_color_equality_per_class(x_data,y_data)
        print(rasp)
        res = shift_columns(rasp)
        # print(res)
        # print(res[:4])
        for i in res:
            print(query(i))

# count how many times R/G/B is the biggest in the image samples, count all times equal, count no times equal
# count how many image have a sum of R/G/B values higher comparatively to others
#4x bin 0-63, 64-127..?
