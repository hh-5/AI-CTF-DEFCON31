from time import sleep

import numpy as np
import string
from PIL import Image, ImageDraw, ImageFont
import requests

from database_utils import insert_image_and_response, get_response_for_image, setup_database
from main_attack import image_to_bytes, numpy_to_image


def query_with_cache(input_image, conn):
    # array_bytes = input_image.tobytes()
    # print(hash(array_bytes))

    # pil_image = numpy_to_image(input_image)
    image_data = image_to_bytes(255-input_image, format="PNG")
    cached_logits = get_response_for_image(conn, image_data)

    # print(cached_logits)
    if cached_logits is None:
        input_array = np.array(input_image).reshape(1, 32, 32, 1)
        response = requests.post('http://inversion.advml.com/score', json={'data': input_array.tolist()})
        logits = response.json()
        logits = logits['outputs'][0]

        # print(f"Response: {logits}")
        pil_image = numpy_to_image(input_image)
        image_data = image_to_bytes(pil_image, format="PNG")
        insert_image_and_response(conn, image_data, logits)
    else:
        logits = list(cached_logits)

    return logits
# Function to check if a character fits in the image bounds
def character_fits(draw, char, font, x, y):
    text_width = draw.textlength(char, font=font)
    return x + text_width <= 32 and y + font.size <= 32


# Function to generate an image with a character at a specific x,y position
def generate_character_image(char, x, y):
    img = np.full((32, 32), 0, dtype=np.uint8)  # Base color
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 32)  # Adjust font size if needed
    except IOError:
        font = ImageFont.load_default()

    if character_fits(draw, char, font, x, y):
        draw.text((x, y), char, font=font, fill=255)
        return np.array(img_pil) / 255.0
    return None


def generate_character_image(char, x, y, fontSize):
    img = np.full((32, 32), 0, dtype=np.uint8)  # Base color
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", fontSize)  # Adjust font size if needed
    except IOError:
        font = ImageFont.load_default()

    if character_fits(draw, char, font, x, y):
        draw.text((x, y), char, font=font, fill=255)
        return np.array(img_pil)
    return None


conn, _ = setup_database()
# Iterating over all printable characters and x,y positions
# for char in "[]{}|\\#$%&'!\"()*+,-./:;<=>?@^_`~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":  # Exclude non-printable characters
#     for x in range(32):
#         for y in range(32):
#             img = generate_character_image(char, x, y,24)
#             if img is not None:
#                     retries = 10
#                     for _ in range(retries):
#                         try:
#                             query_with_cache(img, conn)
#                             break  # If successful, break out of the loop
#                         except:
#                             if _ < retries - 1:  # No need to sleep after the last attempt
#                                 sleep(30)


conn, _ = setup_database()
# Iterating over all printable characters and x,y positions
for i in range(8,32):
    for char in ",.'`\":;-[]{}|<>*+@~^()$%&":#\\#$%&!()*+/<=>?@^_~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":  # Exclude non-printable characters
        for x in range(32):
            for y in range(32):
                img = generate_character_image(char, x, y, i)
                if img is not None:
                    retries = 10
                    for _ in range(retries):
                        try:
                            query_with_cache(img, conn)
                            break  # If successful, break out of the loop
                        except:
                            if _ < retries - 1:  # No need to sleep after the last attempt
                                sleep(30)
import numpy as np
from emnist import extract_training_samples, extract_test_samples

def process_emnist_data(dataset='letters'):
    """
    Fetches the EMNIST dataset for the given category and processes it.
    :param dataset: 'letters', 'digits', or 'symbols'.
    :return: processed data
    """
    if dataset == 'letters':
        images, labels = extract_training_samples('letters')
    elif dataset == 'digits':
        images, labels = extract_training_samples('digits')
    elif dataset == 'symbols':
        images, labels = extract_training_samples('balanced')  # EMNIST does not have a "symbols" only dataset, so we use balanced as an example.
    else:
        raise ValueError("Invalid dataset type. Choose from 'letters', 'digits', or 'symbols'.")

    # Normalize and reshape the images
    images = images.astype('float32') / 255
    images = images.reshape(images.shape[0], 28, 28, 1)

    return images, labels

# For demonstration purposes:
letters_images, letters_labels = process_emnist_data('letters')
symbols_images, symbols_labels = process_emnist_data('symbols')
digits_images, digits_labels = process_emnist_data('digits')


def process_and_query_emnist(conn, dataset='letters'):
    """
    Fetches the EMNIST dataset for the given category, processes it, and queries with cache.
    :param conn: Database connection.
    :param dataset: 'letters', 'digits', or 'symbols'.
    """
    if dataset == 'letters':
        images, labels = extract_training_samples('letters')
    elif dataset == 'digits':
        images, labels = extract_training_samples('digits')
    elif dataset == 'symbols':
        images, labels = extract_training_samples('balanced')  # Handle the "symbols" case
    else:
        raise ValueError("Invalid dataset type. Choose from 'letters', 'digits', or 'symbols'.")

    # Resizing images from 28x28 to 32x32 and normalizing
    processed_images = []
    for img in images:
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((32, 32), Image.BILINEAR)
        processed_images.append(np.array(pil_img))

    for img in processed_images:
        query_with_cache(img, conn)


# For demonstration purposes:
# process_and_query_emnist(conn, 'symbols')




