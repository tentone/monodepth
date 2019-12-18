import numpy as np
from skimage.transform import resize

def depth_norm(x, max_depth):
    return max_depth / x

def predict(model, image, minDepth=10, maxDepth=1000, batch_size=2):
    # Grayscale image
    if len(image.shape) < 3:
        image = np.stack((image, image, image), axis=2)

    # RGB image
    if len(image.shape) < 4:
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # Compute predictions
    predictions = model.predict(image, batch_size=batch_size)

    # Put in expected range
    return np.clip(depth_norm(predictions, max_depth=1000), minDepth, maxDepth) / maxDepth

def scale_up(scale, images):
    scaled = []
    
    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append(resize(img, output_shape, order=1, preserve_range=True, mode="reflect", anti_aliasing=True))

    return np.stack(scaled)

def to_multichannel(i):
    if i.shape[2] == 3: return i
    i = i[:,:,0]
    return np.stack((i,i,i), axis=2)

def evaluate(model, rgb, depth, crop, batch_size=6, verbose=False):
    # Error computaiton based on https://github.com/tinghuiz/SfMLearner
    def compute_errors(gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()

        return a1, a2, a3, abs_rel, rmse, log_10

    depth_scores = np.zeros((6, len(rgb))) # six metrics

    bs = batch_size

    for i in range(len(rgb)//bs):    
        x = rgb[i * bs:(i + 1) * bs, :, :, :]
        
        # Compute results
        true_y = depth[i * bs:(i + 1) * bs, :, :]
        pred_y = scale_up(2, predict(model, x/255, minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0
        
        # Test time augmentation: mirror image estimate
        pred_y_flip = scale_up(2, predict(model, x[...,::-1,:]/255, minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0

        # Crop based on Eigen et al. crop
        true_y = true_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_y = pred_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_y_flip = pred_y_flip[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        
        # Compute errors per image in batch
        for j in range(len(true_y)):
            errors = compute_errors(true_y[j], (0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j])))
            
            for k in range(len(errors)):
                depth_scores[k][(i*bs)+j] = errors[k]

    e = depth_scores.mean(axis=1)

    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format("a1", "a2", "a3", "rel", "rms", "log_10"))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))

    return e
