import tensorflow as tf
import numpy as np
from matplotlib.pyplot import imshow
from PIL import Image
import imageio
from tensorflow import keras

from utils import *
from vgg19 import *
from cost import *

def style_transfer(content_path, style_path, num_iter=200):
    vgg19 = VGG19()
    model = vgg19.load_vgg_model()

    content_image = imageio.imread(content_path)
    content_image = reshape_normalise(content_image)

    style_image = imageio.imread(style_path)
    style_image = reshape_normalise(style_image)

    generated_image = generate_noise_image(content_image)

    # assign content and style image as an input to vgg model
    f_content = tf.function(model['input'].assign(content_image))
    tf.function(model['input'].assign(style_image))

    # selecting output tensorlayer - neither to deep nor to shallow
    out = model['conv4_2']

    aC = f_content(out)
    aG = out

    # computing various costs
    cost = Cost()
    J_content = cost.compute_content_cost(aC, aG)
    J_style = cost.compute_style_cost(model)
    J = cost.total_cost(J_content, J_style)

    # defining optimizer
    lr = 2.0
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    # defining training step
    train_step = optimizer.minimize(J)


    # training a model
    tf.function(model['input'].assign(generated_image))

    for i in range(num_iter):
        # running seesion on train_step to minmize total cost
        tf.function(train_step)

        generated_image = tf.function(generated_image)

        # printing after every 20 iteration
        if i%20 == 0:
            Jt,Jc, Js = tf.function([J, J_content, J_style])
            print("Iteration" + str(i) + " :")
            print("Total Cost" + str(Jt))
            print("Content Cost" + str(Jc))
            print("Style Cost" + str(Js))

            # save the current generated image in the output directory
            save_image(str(i),generated_image)

    # save the final generated image 
    save_image("final",generated_image)

    return generated_image

    