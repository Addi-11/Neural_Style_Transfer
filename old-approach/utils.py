import numpy as np
import imageio
from matplotlib.pyplot import imshow


class CONFIG:
	IMAGE_WIDTH = 400
	IMAGE_HEIGHT = 300
	COLOR_CHANNELS = 3
	NOISE_RATIO = 0.6
	MEAN_PIXEL = np.array([123.68,116.779, 103.939]).reshape((1,1,1,3))
	OUTPUT_DIR = 'output/'
	PRE_TRAINED_PATH = 'pre_trained_model/imagenet-vgg-verydeep-19.mat'


def generate_noise_image(content_image, noise_ratio = CONFIG.NOISE_RATIO):
	"""
	Generates a noisy image by adding noise to the content_image
	"""
	noise_img = np.random.uniform(-20,20,(1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)).astype('float32')

	# Setting the resulting image to be the weighted average of the content image and noise_image
	result_img = noise_img * noise_ratio + content_image * (1 - noise_ratio)

	return result_img

def reshape_normalise(img):
	"""
	Reshape and normalize the input image (content or style)
	"""
	# The image shape is expected to match the input of VGG19
	img = np.resize(img, (1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)).astype('float32')
	img -= CONFIG.MEAN_PIXEL
	return img

def undo_normalise(img):
	"""
	Un-normalise so that the image looks appealing
	"""
	return img + CONFIG.MEAN_PIXEL

def plot_image(img):
	imshow(img)

def save_image(path, image):
	undo_normalise(image)
	# clip and save image
	image = np.clip(image[0], 0, 255).astype('uint8')
	imageio.imwrite("output/"+path+".png", image)