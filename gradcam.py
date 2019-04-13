import numpy as np
import cv2

def gradcam(conv_activations, conv_gradient, guided_backprop):
    '''Guided-Gradcam algorithm.

    Args:
        conv_activations (np.float32): activations of each filter on the last convolutional layer [filter_height, filter_width, last_conv_layer_depth]
        conv_gradient (np.float32): gradient of the target logit w.r.t. each filter on the last convolutional layer [filter_height, filter_width, last_conv_layer_depth]
        guided_backprop (np.float32): gradient of the target logit w.r.t. the input image [image_height, image_width, num_img_channels]
    Returns:
        gradcam (np.float32): GradCAM map [image_height, image_width]
        guided_backprop (np.float32): gradient of the target logit w.r.t. the input image [image_height, image_width, num_img_channels]
        guided_gradcam (np.float32): Guided GradCAM map [image_height, image_width, num_img_channels]
    '''

    #get image size from guided_backprop gradient
    image_size = (guided_backprop.shape[0], guided_backprop.shape[1])

    #gradcam
    weights = np.mean(conv_gradient, axis=(0, 1))
    
    gradcam = np.zeros((conv_activations.shape[0], conv_activations.shape[1]), dtype = np.float32)
    for i, w in enumerate(weights):
        gradcam += w * conv_activations[:, :, i]

    #relu
    gradcam = np.maximum(gradcam, 0)
    
    #normalize and resize to original image size
    gradcam = gradcam / gradcam.max() 
    gradcam = cv2.resize(gradcam, image_size)

    guided_backprop -= guided_backprop.min()  
    guided_backprop /= guided_backprop.max()

    guided_gradcam = np.dstack((np.multiply(guided_backprop[:,:,0],gradcam),
                                np.multiply(guided_backprop[:,:,1],gradcam),
                                np.multiply(guided_backprop[:,:,2],gradcam)))

    guided_gradcam -= guided_gradcam.min()  
    guided_gradcam /= guided_gradcam.max()

    return gradcam, guided_backprop, guided_gradcam