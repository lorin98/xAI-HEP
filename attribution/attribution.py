import numpy as np
import tensorflow as tf

def ram_per_feature(ram_model,
                  model,
                  image,
                  ft_index: int # feature index = 0 for pt, 1 for eta
                  ):

    assert ft_index == 0 or ft_index == 1, "Unexpected value for ft_index"
    # extracting features from the last conv layer and the results from the last dense layer
    features, _ = ram_model.predict(np.expand_dims(image, axis=0))

    ram_features = features[0]
    # getting the weights from the last dense layer, in particular the weights
    # corresponding to the node responsible for the specified feature
    ram_weights = model.layers[-3].get_weights()[0][:,ft_index]
    # dot product to obtain the heatmap
    ram_output = np.dot(ram_features, ram_weights)

    return ram_output

def interpolate_image(image):
    rows, cols = np.where(image == 1) # get the position of the light pixels
    n_interpolations = len(rows) + 1  # number of light pixels + baseline at the beginning

    interpolated_images = np.empty(shape=(n_interpolations, image.shape[0], image.shape[1]), dtype=np.int8)

    baseline = np.zeros_like(image)
    interpolated_images[0] = baseline

    # iteratively adding new light pixels one by one
    for i, (r, c) in enumerate(zip(rows, cols)):
        baseline[r, c] = 1
        interpolated_images[i+1] = baseline

    # convert to tensors for next step
    interpolated_images = tf.convert_to_tensor(interpolated_images, dtype=tf.float32)
    return interpolated_images

def integrated_gradients(model, images):
    # compute gradients
    with tf.GradientTape() as tape:
        tape.watch(images)
        pred = model(images)
    path_gradients = tape.gradient(pred, images)

    # accumulate gradients with riemann_trapezoidal
    grads = (path_gradients[:-1] + path_gradients[1:]) / tf.constant(2.0)
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    return integrated_gradients

def add_noise(image, noise_percent):
    noisy_image = image.copy()
    rows, cols = image.shape

    cnt = 0
    for r in range(rows):
        for c in range(cols):
            if image[r, c] == 0 and np.random.random() < noise_percent:
                noisy_image[r, c] = 1
                cnt += 1

    noised_pixels = cnt / (rows * cols)

    return noisy_image, noised_pixels

def smoothgrad(model, image, n=100, noise_percent=0.02):
    sg = np.zeros_like(image)
    for _ in range(n):
        noisy_image = add_noise(image, noise_percent)[0]
        interpolated_images = interpolate_image(noisy_image)
        ig = integrated_gradients(model, interpolated_images)
        sg += ig
    sg /= n
    return sg

def get_heatmaps(model, ram_model, image):
    ram_pt = ram_per_feature(ram_model, model, image, 0)
    ram_eta = ram_per_feature(ram_model, model, image, 1)

    interpolated_images = interpolate_image(image)
    ig = integrated_gradients(model, interpolated_images)
    sg = smoothgrad(model, image)

    return ram_pt, ram_eta, ig, sg
