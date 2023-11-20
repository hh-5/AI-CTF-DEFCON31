import io
import os
import random
import string
import time

import torch
from keras.datasets import cifar100
import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image, ImageDraw, ImageFont
from art.attacks.inference.model_inversion.mi_face import MIFace
from art.estimators.classification import KerasClassifier

from tensorflow.keras.optimizers import Adam
from database_utils import setup_database, close_database, insert_image_and_response, get_response_for_image
from surrogate_model import create_surrogate_model, save_surrogate_model
import tensorflow as tf

# tf.compat.v1.disable_eager_execution()
tf.config.experimental_run_functions_eagerly(True)
tf.executing_eagerly()

min458 = 1e7

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    raise RuntimeError("No GPU found!")


# tf.enable_eager_execution()
def numpy_to_image(np_array):
    # Convert numpy array to PIL Image
    return Image.fromarray((np_array * 255).astype(np.uint8).squeeze())


def image_to_bytes(img, format="PNG"):
    if isinstance(img, np.ndarray):
        img = Image.fromarray((img).astype(np.uint8))  # Convert ndarray to PIL Image

    with io.BytesIO() as output:
        img.save(output, format=format)
        return output.getvalue()


def query_with_cache(input_image, conn):
    # array_bytes = input_image.tobytes()
    # print(hash(array_bytes))

    # diffx = random.randint(-10,10)
    # diffy = random.randint(-10,10)
    # input_image[:10+diffx, :, :] = 0
    # input_image[21+diffx:, :, :] = 0
    # input_image[:, :10+diffy, :] = 0
    # input_image[:, 21+diffy:, :] = 0
    pil_image = numpy_to_image(1-input_image)
    image_data = image_to_bytes(pil_image, format="PNG")
    cached_logits = get_response_for_image(conn, image_data)

    # print(cached_logits)
    if cached_logits is None:
        input_array = np.array(input_image).reshape(1, 32, 32, 1) / -1
        # print(input_array)

        response = requests.post('http://inversion.advml.com/score', json={'data': input_array.tolist()})
        logits = response.json()
        logits = logits['outputs'][0]

        # print(f"Response: {logits}")
        pil_image = numpy_to_image(input_image)
        # print(input_image)
        image_data = image_to_bytes(pil_image, format="PNG")
        insert_image_and_response(conn, image_data, logits)
    else:
        logits = list(cached_logits)

    return logits


(x_train_global, y_train_global), (_, _) = cifar100.load_data(label_mode='fine')


def get_class_indices(y):
    class_indices = {}
    for i in range(100):  # CIFAR-100 has 100 classes
        class_indices[i] = np.where(y == i)[0]
    return class_indices


class_indices_global = get_class_indices(y_train_global)

class_counter_global = 0


# def generate_image_v2(num_samples):
#     global x_train_global, class_indices_global, class_counter_global
#
#     images = []
#     for _ in range(num_samples):
#         current_class_indices = class_indices_global[class_counter_global % 100]  # Adjusted for 100 classes
#
#         # Randomly pick an image from the current class
#         random_index = np.random.choice(current_class_indices)
#         img = x_train_global[random_index]
#
#         # Convert the image to grayscale
#         gray_img = np.dot(img[..., :3], [0.299, 0.587, 0.114])
#         gray_img = gray_img.astype(np.uint8)
#         images.append(gray_img.reshape(32, 32, 1))
#
#         # Move to the next class
#         class_counter_global += 1
#
#     return np.array(images)


def generate_image_v2(num_samples):
    images = []

    for _ in range(num_samples):
        # 1. Choose base color
        base_color_val = 0  # np.random.choice([0])  # white, light gray, dark gray, black
        img = np.full((32, 32), base_color_val, dtype=np.uint8)

        # 2. Decide on the content to add
        content_choice = np.random.choice(['char', 'rect', 'random'], p=[1, 0, 0])

        if content_choice == 'char':
            char = random.choice(string.printable[:-5])  # Exclude non-printable characters

            # Randomly select character color and background color
            char_color = 255# - random.randint(0,10)

            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)

            # Set font size
            font_size = np.random.randint(5, 30)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()

            # Get text width and height

            text_width, text_height = draw.textlength(char, font=font), font.getmetrics()[1]

            # Set random x, y position of the text ensuring it's fully visible
            x = np.random.randint(0, int(32 - text_width))  # ???
            y = np.random.randint(0, 32- font_size)
            draw.text((x, y), char, font=font, fill=char_color)

            img = np.array(img_pil)

        elif content_choice == 'rect':
            # Random rectangle region
            x1, x2 = sorted(np.random.randint(16, 32, 2))
            y1, y2 = sorted(np.random.randint(16, 32, 2))

            percent_cover = np.random.uniform(0.3, 0.7)
            num_pixels = int((x2 - x1) * (y2 - y1) * percent_cover)

            for _ in range(num_pixels):
                x = np.random.randint(x1, x2)
                y = np.random.randint(y1, y2)
                if random.random() < 0.15:
                    img[y, x] = random.randint(150, 255)
                else:
                    img[y, x] = 255

        else:  # content_choice == 'random'
            num_pixels = np.random.randint(1, 10)
            for _ in range(num_pixels):
                x = np.random.randint(0, 32)
                y = np.random.randint(0, 32)
                if random.random() < 0.15:
                    img[y, x] = random.randint(130, 255)
                else:
                    img[y, x] = 255

        images.append(img)

    return np.array(images).reshape(num_samples, 32, 32, 1)

main_graph = tf.Graph()

import torch.optim as optim


def generate_adversarial_images(model, class_id, num_samples, ALPHA, NUM_GRAD_ASCENT_STEPS):
    # Assuming the generate_image_v2 function provides random images of shape (num_samples, 32, 32, 1)
    # def generate_image_v2(samples):
    #     return np.random.rand(samples, 32, 32, 1) * 255

    # Generate initial image samples
    img_samples = generate_image_v2(num_samples)

    # Clip and convert the initial images
    img_samples = np.clip(img_samples / 255, 0, 1).astype(np.float32)

    images = tf.Variable(img_samples, dtype=tf.float32, trainable=True)
    # original_images = tf.identity(images).numpy()

    initial_predictions = model(images)
    initial_confidence = initial_predictions[:, class_id].numpy()
    print(f"Initial confidence for class {class_id:.5f}:", initial_confidence)
    target = random.choice([4, 5, 7])
    for step in range(NUM_GRAD_ASCENT_STEPS):
        with tf.GradientTape() as tape:
            tape.watch(images)

            # Obtain the model's predictions for the images
            predictions = model(images)

            # Using tf.gather to obtain specific class probabilities
            # minimize_probs = tf.gather(predictions, [0, 1, 2, 3, 6], axis=1)
            # minimize_sum = tf.reduce_sum(minimize_probs, axis=1)

            maximize_sum = predictions[:, class_id]

            # Combine the objectives: maximize (maximize_sum - minimize_sum)
            loss = tf.reduce_mean(maximize_sum)  # - minimize_sum)
        # for step in range(NUM_GRAD_ASCENT_STEPS):
        #     with tf.GradientTape() as tape:
        #         tape.watch(images)
        #         predictions = model(images)
        #         if class_id in [4, 5, 7]:
        #             loss = tf.reduce_mean(predictions[:, class_id])  # negative for maximization
        #         else:
        #             # Minimize the maximum confidence for classes [0,1,2,3,6]
        #             selected_predictions = tf.gather(predictions, [0, 1, 2, 3, 6], axis=1)
        #             loss = tf.reduce_mean(selected_predictions)
        # Compute the gradients of the loss with respect to the images
        grads = tape.gradient(loss, images)

        # Normalize the gradients
        # grads = tf.math.l2_normalize(grads)

        # Update the images using the gradients and the step size
        images = images + tf.sign(grads) * ALPHA

        # Optionally: Clip the images to ensure they stay in a valid range, e.g., [0, 1]
        images = tf.clip_by_value(images, 0, 1)

        # Evaluate the model's predictions after the update
        updated_predictions = model(images)
        updated_confidence = updated_predictions[:, class_id].numpy()
        # print(f"Step {step + 1}, Confidence for class {class_id}: {updated_confidence[0]:.5f}")

    # Predict on perturbed images and get the confidence for the target class
    final_predictions = model(images)
    final_confidence = final_predictions[:, class_id].numpy()
    print(f"Final confidence for class {class_id:.5f} after perturbation:", final_confidence)

    # Ensure the values are clipped between 0 and 255
    generated_imgs = np.clip(images.numpy() * 255, 0, 255).astype(np.float32)

    return generated_imgs


# with main_graph.as_default():
#     images = tf.Variable(tf.zeros([4, 32, 32, 1]), dtype=tf.float32)  # numsamples

def generate_adversarial_images(model, class_id, num_samples, ALPHA, NUM_GRAD_ASCENT_STEPS, grayscale_range=(0, 1)):
    # Helper function to generate random images
    # def generate_image_v2(samples):
    #     return np.random.rand(samples, 32, 32, 1) * 255

    # Generate initial image samples
    img_samples = generate_image_v2(num_samples)
    img_samples = np.clip(img_samples / 255.0, -1, 1).astype(np.float32)

    images = tf.Variable(img_samples, dtype=tf.float32, trainable=True)

    # Initial predictions to print initial confidences
    # initial_predictions = model(images)
    # initial_confidence = initial_predictions[:, class_id].numpy()
    # print(f"Initial confidence for class {class_id}:", initial_confidence)

    optimizer = tf.optimizers.Adam(learning_rate=ALPHA)

    diffx = random.randint(-8, 8)
    diffy = random.randint(-8, 8)
    # for step in range(NUM_GRAD_ASCENT_STEPS):
    #     with tf.GradientTape() as tape:
    #         tape.watch(images)
    #         predictions = model(images)
    #         maximize_probs = tf.reduce_mean(tf.gather(predictions, [class_id], axis=1))
    #         minimize_probs = tf.reduce_mean(tf.gather(predictions, [0, 1, 2, 3, 6], axis=1))
    #         loss = -maximize_probs + minimize_probs
    #
    #     # Compute the gradient of the loss with respect to the full images
    #     grads = tape.gradient(loss, images)
    #
    #     # Convert the images tensor to a numpy array
    #     image_array = images.numpy()
    #     grads_array = grads.numpy()
    #
    #     # for i in range(num_samples):
    #     #
    #     #     patch_start_x = 8 + diffx
    #     #     patch_end_x = 24 + diffx
    #     #     patch_start_y = 8 + diffy
    #     #     patch_end_y = 24 + diffy
    #     #
    #     #     # Apply the patch gradients to the image numpy array
    #     #     image_array[i, patch_start_x:patch_end_x, patch_start_y:patch_end_y, :] += ALPHA * grads_array[i,
    #     #                                                                                        patch_start_x:patch_end_x,
    #     #                                                                                        patch_start_y:patch_end_y,
    #     #                                                                                        :]
    #     #
    #     #     # Zero out other regions
    #     #     image_array[i, :patch_start_x, :, :] = 0
    #     #     image_array[i, patch_end_x:, :, :] = 0
    #     #     image_array[i, :, :patch_start_y, :] = 0
    #     #     image_array[i, :, patch_end_y:, :] = 0
    #
    #     # Convert the numpy array back to a tensor
    #     images = tf.constant(image_array)

    # Final predictions to print confidences after perturbations
    # final_predictions = model(images)
    # final_confidence = final_predictions[:, class_id].numpy()
    # print(f"Final confidence for class {class_id} after perturbation:", final_confidence)

    # Convert images to the desired grayscale range and return
    normalized_images = grayscale_range[0] + images.numpy() * (grayscale_range[1] - grayscale_range[0])
    generated_imgs = np.clip(normalized_images, grayscale_range[0], grayscale_range[1]).astype(np.float32)
    return generated_imgs


def run_attack_and_plot():
    conn, _ = setup_database()

    MAX_ITER = 1000111
    CHECKPOINT_FREQ = 1
    BATCH_SIZE = 32
    NUM_SAMPLES_PER_CLASS = 32
    b457 = 1e16

    for iteration in range(MAX_ITER):
        start_time = time.time()

        # surrogate = create_surrogate_model()
        print("Good weights")
        lr = 0.002

        # optimizer = Adam(learning_rate=lr)

        # surrogate.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        print(f"\nIteration {iteration} started")

        # Step 1: Generate Class-specific Images
        adversarial_start_time = time.time()
        generated_images = []
        for class_id in [4, 5, 7]:
            images_for_class = generate_adversarial_images(None, class_id, NUM_SAMPLES_PER_CLASS, 0.05, 5)
            generated_images.append(images_for_class)
        print(f"Adversarial image generation time: {time.time() - adversarial_start_time} seconds")

        generated_images_array = np.concatenate(generated_images, axis=0)

        # Step 2: Query the Actual System
        query_start_time = time.time()
        api_predictions = [query_with_cache(sample, conn) for sample in generated_images_array]
        print(f"API query time: {time.time() - query_start_time} seconds")
        training_start_time = time.time()

        api_predictions_array = np.array(api_predictions)
        p457 = api_predictions_array[:, [4, 5, 7]]
        reciprocal_value = 1 / np.max(p457)
        if reciprocal_value < b457:
            b457 = reciprocal_value
        print(f"Current 457 best at {b457}")

        # Replace 1e7 with the reciprocal value for positions 4, 5, and 7
        # weights = [1, 1, 1, 1, b457, b457, 0.1, b457]
        # # Step 3: Train the Surrogate Model
        # class_weight = {i: weights[i] for i in range(8)}
        # surrogate.fit(generated_images_array, api_predictions_array, batch_size=BATCH_SIZE, epochs=1,
        #               class_weight=class_weight, verbose=1)
        # print(f"Surrogate training time: {time.time() - training_start_time} seconds")

        # Step 4: Use the surrogate model for the attack
        # attack_start_time = time.time()
        # surrogate_classifier = KerasClassifier(model=surrogate, clip_values=(0, 255))
        # attack = MIFace(surrogate_classifier, max_iter=1, threshold=1.)
        # x_inferred = attack.infer(None, y)
        # print(f"Attack time: {time.time() - attack_start_time} seconds")
        #
        # # Visualization after each iteration
        # if (iteration + 1) % 1 == 0 or iteration == 0:
        #     plt.figure(figsize=(20, 10))
        #     for i in range(8):
        #         plt.subplot(2, 4, i + 1)
        #         plt.imshow(np.reshape(x_inferred[i], (32, 32)), cmap=plt.cm.gray_r)
        #         plt.axis('off')
        #
        #     plt.suptitle(f"Results after {iteration} iterations for initialization")
        #     plt.tight_layout()
        #
        #     # Save the plot to an image file
        #     output_directory = "saved_plots"
        #     if not os.path.exists(output_directory):
        #         os.makedirs(output_directory)
        #     output_path = os.path.join(output_directory, f"Iteration_{iteration + 15}.png")
        #     plt.savefig(output_path)
        #     plt.close()
        #
        # Save the surrogate model every CHECKPOINT_FREQ epochs
        # if (iteration + 1) % CHECKPOINT_FREQ == 0:
        #     save_surrogate_model(surrogate, iteration + 1)

        print(f"Total time for iteration {iteration}: {time.time() - start_time} seconds")

    close_database(conn)


if __name__ == "__main__":
    run_attack_and_plot()

