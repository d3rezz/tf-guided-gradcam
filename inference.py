import numpy as np
import os
import glob
import re
import cv2
import tensorflow as tf
from tensorflow.contrib import slim
from models.research.slim.nets import mobilenet_v1
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from gradcam import gradcam

DEPTH_MULTIPLIER=1.0
IMAGE_SIZE=[224, 224]
CHECKPOINT='mobilenet_v1_1.0_224.ckpt'
NUM_CLASSES = 1001


# Guided Relu
# In addition to propagating the error back to every positive input, only propagate back positive error signals
# See https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb
@ops.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    return tf.where(0. < grad, gen_nn_ops.relu_grad(grad, op.outputs[0]), tf.zeros_like(grad))


def mobilenet_preprocess(image, image_size=IMAGE_SIZE):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    #Resize image
    image = tf.image.resize_images(image, image_size)   #bilinear resize, will distort if aspect ratio is not the same

    return image


def build_model(images):
    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
        logits, endpoints = mobilenet_v1.mobilenet_v1(images,
                                                            num_classes=NUM_CLASSES,
                                                            is_training=False,
                                                            depth_multiplier=DEPTH_MULTIPLIER)
    return logits, endpoints


def get_init_fn_last_checkpoint(checkpoint_path):
    if os.path.isdir(checkpoint_path):
        #load the most recent checkpoint in the provided directory
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    current_gs = 0
    matches = re.search(r".ckpt-([0-9]+)", checkpoint_path, re.IGNORECASE)
    if matches:
        current_gs = int(matches.group(1))

    variables_to_restore = slim.get_variables_to_restore()

    init_fn = slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore)

    return init_fn, current_gs



def main(argv=None):
    tf.logging.set_verbosity(tf.logging.INFO)

    #Load image
    img = cv2.imread(FLAGS.image)

    #convert to float32
    img = cv2.normalize(img.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)

    #convert to rgb
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #load imagenet labels for displaying predicted class
    labels=[]
    with open("labels.txt", "r") as f:
        for line in f:
            labels.append(line)

    graph = tf.Graph()
    with graph.as_default():
        with graph.gradient_override_map({'Relu6': 'GuidedRelu'}):   #Replace every Relu6 with a GuidedRelu
            
            image = tf.placeholder(tf.float32, shape=[None, None, 3], name="input_image")
            processed_image = mobilenet_preprocess(image, IMAGE_SIZE)
            processed_images = tf.expand_dims(processed_image, 0)

            #Load model
            logits, endpoints = build_model(processed_images)
            max_logit = tf.reduce_max(logits, axis=1)   #Get value of the highest logit (logit of predicted class)
            predictions = tf.argmax(logits, axis=1)
            conv_activations = endpoints['Conv2d_13_pointwise']

            #If image label is not provided, use prediction
            conv_gradient = None
            if FLAGS.label is None:
                conv_gradient = tf.gradients(max_logit, conv_activations)[0]    #tf.gradient returns a list and we only want the gradient of 1 tensor
            else:
                conv_gradient = tf.gradients(logits[:, FLAGS.label], conv_activations)[0]

            if FLAGS.label is None:
                guided_backprop =tf.gradients(max_logit, processed_images)[0]
            else:
                guided_backprop = tf.gradients(logits[:, FLAGS.label], processed_images)[0]

            #Op to load model checkpoint
            init_fn, current_gs = get_init_fn_last_checkpoint(CHECKPOINT)

    with tf.Session(graph=graph) as sess:
        #Initializations
        init_fn(sess)

        pred, conv_acts, conv_grad, gbp = sess.run([predictions, conv_activations, conv_gradient, guided_backprop], feed_dict={image: img_rgb})

        #Show Original image and predicted class
        resized_img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]))

        if FLAGS.label is None:
            label_string = labels[pred[0]]
            cv2.imshow("Predicted:%s %d"%(label_string, pred), resized_img)
        else:
            label_string = labels[FLAGS.label]
            cv2.imshow("Target:%s %d"%(label_string, pred), resized_img)

        #batch should contain only first image so reduce that dimension
        conv_acts = np.squeeze(conv_acts, axis=0)
        conv_grad = np.squeeze(conv_grad, axis=0)
        gbp = np.squeeze(gbp, axis=0)


        gc, gbp, ggc = gradcam(conv_acts, conv_grad, gbp)
        
        #Display and save resulting maps
        gc_heatmap = cv2.applyColorMap(np.uint8(255*gc), cv2.COLORMAP_JET)
        cv2.imshow("Gradcam", gc_heatmap)
        cv2.imshow("Guided backprop", gbp)
        cv2.imshow("Guided Gradcam", ggc)
        cv2.imwrite("gradcam.jpg", gc_heatmap)
        gbp = np.uint8(gbp*255.0)
        ggc = np.uint8(ggc*255.0)
        cv2.imwrite("gbp.jpg", gbp)
        cv2.imwrite("ggc.jpg", ggc)

        cv2.waitKey(0)


if __name__ == '__main__':
    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_string("image",None,
                               """Path to image""")
    tf.app.flags.DEFINE_integer("label",None,
    """Truth label (integer)""")
    tf.app.flags.mark_flag_as_required("image")
    tf.app.run()
