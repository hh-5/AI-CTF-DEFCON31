import keras
import numpy as np
import tensorflow as tf
from art.attacks.inference.model_inversion import MIFace
from art.estimators.classification import KerasClassifier
from keras import backend as K

import os
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from matplotlib import pyplot as plt
from numpy import arange

import os

for filename in os.listdir('.'):
    if filename.endswith('.h5'):
        classifier = keras.models.load_model(filename)
        print(filename)
        print("\n\n")
        # Wrap the Keras model with ART's KerasClassifier
        y = arange(8)
        x_init_white = np.zeros((8, 32, 32, 1))
        x_init_grey = np.zeros((8, 32, 32, 1)) + 0.5
        x_init_black = np.ones((8, 32, 32, 1))
        x_init_random = np.random.uniform(0, 1, (8, 32, 32, 1))

        starts=[x_init_white,x_init_black,x_init_grey,x_init_random]

        for idrx in range(len(starts)):
            start = starts[idrx]
            # Setup the classifier and attack
            surrogate_classifier = KerasClassifier(model=classifier, clip_values=(0, 255))
            attack = MIFace(classifier=surrogate_classifier, max_iter=1000, window_length=100, threshold=0.99,
                            learning_rate=0.1, batch_size=1, verbose=True)
            x_inferred = attack.infer(start, y)
            # print(f"Attack time: {time.time() - now}")
            # now = time.time()

            # Visualization after each iteration
            # if (iteration + 1) % 1 == 0 or iteration == 0:
            plt.figure(figsize=(40, 20))  # Adjusted the figure size
            for i in range(8):
                plt.subplot(2, 4, i + 1)  # Changed to a 2x4 layout
                plt.imshow(np.reshape(x_inferred[i], (32, 32)))
                plt.axis('off')  # Optional: Turns off axis numbering for cleaner display
            plt.suptitle(f" {filename} ")
            plt.tight_layout()

            # Display the plot in Jupyter notebooks
            # plt.show()

            # Save the plot to an image file (commented out for now)
            output_directory = "saved_results"
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            output_path = os.path.join(output_directory, f"{filename}_{idrx}.png")
            plt.savefig(output_path)
            plt.close()

