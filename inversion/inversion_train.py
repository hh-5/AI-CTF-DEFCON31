import io
import sqlite3
import numpy as np
import keras
import tensorflow as tf

from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.src.layers import Dropout
import os
import re
import time
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from database_utils import setup_database, close_database, setup_database_with_name
from surrogate_model import create_surrogate_model, save_surrogate_model

# Constants
BATCH_SIZE = 32

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    raise RuntimeError("No GPU found!")
def preprocess_images(images):
    """
    Decode the image strings, convert them to numpy arrays, and reshape them for the model.
    """
    decoded_images = []

    for img_string in images:
        # Convert image bytes to PIL Image
        pil_image = Image.open(io.BytesIO(img_string[0]))

        # Convert PIL Image to numpy array and normalize
        numpy_image = np.array(pil_image) / 255.0

        # Add channel dimension
        numpy_image = numpy_image[..., np.newaxis]

        decoded_images.append(numpy_image)

    return np.array(decoded_images)


# Training the Surrogate Model
def train_surrogate_model(all_classes, class_weights, epochs, name):
    conn, cursor = setup_database_with_name(name[:-1])
    print(name)

    # Fetch all image and response data from the database
    cursor = conn.cursor()
    cursor.execute('SELECT image_data FROM Images')
    images = cursor.fetchall()
    cursor.execute('SELECT class1, class2, class3, class4, class5, class6, class7, class8 FROM Responses')
    responses = cursor.fetchall()
    images, responses = np.array(images), np.array(responses)
    # Preprocess the images
    images = preprocess_images(images)
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(images, responses, test_size=0.2, random_state=42)

    # Filter for first four classes if all_classes is False
    if not all_classes:
        y_train = y_train[:, :4]
        y_test = y_test[:, :4]

    surrogate = create_surrogate_model()

    weights = [1, 1, 1, 1] if not all_classes else class_weights
    class_weight = {i: weights[i] for i in range(len(weights))}
    print(x_train.shape)
    print(y_train.shape)
    surrogate.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=epochs, class_weight=class_weight, verbose=1)

    # Evaluate the model on the test data
    score = surrogate.evaluate(x_test, y_test, verbose=0)
    print(f"only408 Test loss: {score[0]}, Test accuracy: {score[1]}")

    # Save the surrogate model
    filename = f"res7/{name}_L{score[0]:.3f}_A{score[1]:.3f}.h5"

    surrogate.save(filename)

    close_database(conn)


# Main Execution
if __name__ == "__main__":

    filenames = [f for f in os.listdir() if f.endswith('.db')]
    for name in filenames:
        runs = [
            # ([1, 1, 1, 1, 1, 1, 1, 1], 5),
            ([1, 1, 1, 1, 1, 1, 1, 1], 10),
            # ([1, 1, 1, 1, 1, 1, 1, 1], 20),
            # ([1, 1, 1, 1, 1, 1, 0.1, 1], 10),
            # ([1, 1, 1, 1, 0.001, 0.001, 0.1, 0.001], 10),
            # ([1, 1, 1, 1, 0.001, 0.001, 1, 0.001], 10),
            # ([1, 1, 1, 1, 1e3, 1e3, 1, 1e3], 10),
            # ([1, 1, 1, 1, 1e9, 1e9, 1, 1e9], 10),
            # ([10, 10, 10, 10, 1, 1, 1, 1], 10)
        ]
        for i in range(len(runs)):
            train_surrogate_model(True, runs[i][0], runs[i][1], name + f"{i}")

