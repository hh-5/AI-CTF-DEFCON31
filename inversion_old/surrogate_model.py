import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import os
import re
import keras
from keras.src.layers import Dropout, LeakyReLU


def create_surrogate_model():
    # 1. Check the local directory for .h5 files.
    h5_files = [f for f in os.listdir() if f.endswith('.h5')]

    # Sort the files based on the iteration number in their name
    global valu
    valu = 0

    # Sort the files based on the iteration number in their name
    def extract_number(filename):
        global valu
        match = re.search(r'(\d+)', filename)
        if match:
            valu = max(valu, int(match.group(1)))
        return valu

    h5_files.sort(key=extract_number, reverse=True)

    # 2. Find the file with the highest iteration number in its name.
    if h5_files:
        latest_model_file = h5_files[0]

        # 3. Load and return that model if it exists.
        model = keras.models.load_model(latest_model_file)
        print(f"Loaded model from {latest_model_file}")
        return model#,valu
    else:
        # 4. Otherwise, create a new model and return it.
        model = Sequential()

        # First convolutional layer
        model.add(Conv2D(32, kernel_size=(3, 3), kernel_initializer='he_normal', input_shape=(32, 32, 1)))
        model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))  # After second MaxPooling2D layer

        # Second convolutional layer
        model.add(Conv2D(64, kernel_size=(3, 3), kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # model.add(Dropout(0.25))  # After second MaxPooling2D layer

        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))

        # model.add(Dropout(0.25))  # After second MaxPooling2D layer
        model.add(Dense(8, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        print("Created a new model")
        return model#,0


def save_surrogate_model(model, iteration):
    filename = f"train_surrogate_checkpoint_{iteration}.h5"
    model.save(filename)
    return filename