import numpy as np

def depth_norm(x, max_depth):
    return max_depth / x

def predict(model, image, min_depth, max_depth, batch_size):
    # Grayscale image
    if len(image.shape) < 3:
        image = np.stack((image, image, image), axis=2)

    # RGB image
    if len(image.shape) < 4:
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # Compute predictions
    predictions = model.predict(image, batch_size=batch_size)

    # Put in expected range
    return np.clip(depth_norm(predictions, max_depth=max_depth), min_depth, max_depth) / max_depth
