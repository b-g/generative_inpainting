import argparse
import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
from inpaint_model import InpaintCAModel
import glob

g = None
sess = None
input_image = None


def setup(opts):
    FLAGS = ng.Config('inpaint.yml')
    global g
    global sess
    global input_image
    model = InpaintCAModel()
    g = tf.get_default_graph()
    sess = tf.Session(graph=g)
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    input_image = tf.placeholder(tf.float32, shape=(1, opts['height'], opts['width']*2, 3))
    output_image = model.build_server_graph(FLAGS, input_image)
    output_image = (output_image + 1.) * 127.5
    output_image = tf.reverse(output_image, [-1])
    output_image = tf.saturate_cast(output_image, tf.uint8)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = []
    for var in vars_list: 
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(opts['checkpoint_dir'], from_name)
        assign_ops.append(tf.assign(var, var_value))
    sess.run(assign_ops)
    print('Model loaded')
    return output_image


def inpaint(output_image, image, mask):
    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    feed_dict = {input_image: np.concatenate([image, mask], axis=2)}
    with g.as_default():
        result = sess.run(output_image, feed_dict=feed_dict)
    result = result[0][:, :, ::-1]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='', type=str,
                        help='The filename of image to be completed.')
    parser.add_argument('--mask', default='', type=str,
                        help='The filename of mask, value 255 indicates mask.')
    parser.add_argument('--output', default='output.png', type=str,
                        help='Where to write output.')
    parser.add_argument('--checkpoint_dir', default='', type=str,
                        help='The directory of tensorflow checkpoint.')
    args, unknown = parser.parse_known_args()

    opts = {
        'checkpoint_dir': args.checkpoint_dir,
        'width': 680,
        'height': 512
    }
    output_image = setup(opts)
    image = cv2.imread(args.image)
    mask = cv2.imread(args.mask)
    assert image.shape == mask.shape
    result = inpaint(output_image, image, mask)
    cv2.imwrite(args.output, result)