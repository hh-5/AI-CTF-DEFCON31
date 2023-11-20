import requests
from itertools import combinations, product
import numpy as np
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset using Keras
(train_images, _), (test_images, _) = mnist.load_data()

def combine_counts(counts_list1, counts_list2):
    """Combine two lists of grayscale counts into a (256, 2) list."""
    combined = np.column_stack((counts_list1, counts_list2))
    return combined

def count_intensity_occurrences(images):
    """
    Count the total number of occurrences of each intensity (from 0 to 255) across all images.
    """
    intensity_counts = np.zeros(256, dtype=int)

    for intensity in range(256):
        # Sum the occurrences of the intensity across all images
        intensity_counts[intensity] = np.sum(images == intensity)

    return intensity_counts

def count_intensity_appearance(images):
    """
    Count the number of images in which each intensity (from 0 to 255) appears.
    """
    intensity_counts = np.zeros(256, dtype=int)

    for intensity in range(256):
        # Count images where the intensity appears
        intensity_counts[intensity] = np.sum(np.any(images == intensity, axis=(1, 2)))

    return intensity_counts

def generate_counts(dataset_choice, counting_mode):
    """
    Generate counts based on the dataset and counting mode provided.

    Parameters:
    - dataset_choice: 1 for train, 2 for test, 3 for train + test.
    - counting_mode: 1 for counting total occurrences of all pixels, 
                     2 for counting each grayscale from an image once,
                     3 for total count minus the occurrences (e.g., 10000 - value for test).

    Returns:
    - A (256, 1) list with the counts.
    """

    # Based on dataset choice, select the dataset
    if dataset_choice == 1:
        data = train_images
        total_count = 60000
    elif dataset_choice == 2:
        data = test_images
        total_count = 10000
    elif dataset_choice == 3:
        data = np.concatenate((train_images, test_images))
        total_count = 70000
    else:
        raise ValueError("Invalid dataset choice. Choose 1 for train, 2 for test, or 3 for both.")

    # Count based on the counting mode
    if counting_mode == 1:
        counts = count_intensity_occurrences(data)
    elif counting_mode == 2:
        counts = count_intensity_appearance(data)
    elif counting_mode == 3:
        if dataset_choice == 1:
            counts = 60000 - count_intensity_appearance(train_images)
        elif dataset_choice == 2:
            counts = 10000 - count_intensity_appearance(test_images)
        else:
            counts = 70000 - count_intensity_appearance(np.concatenate((train_images, test_images)))
    else:
        raise ValueError("Invalid counting mode. Choose 1 for total occurrences, 2 for grayscale appearance, or 3 for total - count.")

    # Convert the counts to (256, 1) format
    return counts.reshape(256, 1)

def transform_list(input_list):
    if len(input_list) != 256 or any(len(sublist) != 2 for sublist in input_list):
        raise ValueError("Input list should be of size 256x2.")
    original = input_list.copy()
    reversed_list = input_list[::-1]
    swapped = input_list.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    reversed_swapped = swapped[::-1]
    return [original, reversed_list, swapped, reversed_swapped]

lists = [[x for x in range(1, 257)],[x for x in range(0, 256)]]

def query(input_data):
    # Convert numpy arrays to standard Python lists for JSON serialization
    input_data_list = input_data.tolist()

    response = requests.post('http://count-mnist.advml.com/score', json={'data': input_data_list})
    if 'flag' in response or len(response.json()) > 50:
        print (input_data)
    return response.json()
for i in [3]:
    for j in [1]:
        lists.append(generate_counts(i,j))
# print (lists[1])
# print (lists[8])
for i in range(len(lists)):
    for j in range(i+1,len(lists)):
        aux = transform_list(combine_counts(lists[i],lists[j]))
#         print (aux)
        for k in aux:
            print(f"{i} {j} \n{k[0]}\n")
            print(query(k))
