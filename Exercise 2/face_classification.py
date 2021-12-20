import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import random as rnd
import sklearn
from sklearn.datasets import fetch_olivetti_faces
import os
from scipy.fftpack import dct
from numpy import random
from cv2 import cv2


def get_histogram(image, param = 30):
    hist, bins = np.histogram(image, bins=np.linspace(0, 1, param))
    return [hist, bins]

def get_dft(image, mat_side = 13):
    f = np.fft.fft2(image)
    f = f[0:mat_side, 0:mat_side]
    return np.abs(f)

def get_dct(image, mat_side = 13):
    c = dct(image, axis=1)
    c = dct(c, axis=0)
    c = c[0:mat_side, 0:mat_side]
    return c

def get_gradient(image, n = 2):
    n=n-1
    shape = image.shape[0]
    i, l = 0, 0
    r = n
    result = []

    while r <= shape:
        window = image[l:r, :]
        result.append(np.sum(window))
        i += 1
        l = i * n
        r = (i + 1) * n
    result = np.array(result)
    return result

def get_scale(image, scale = 0.35):
	#image = image.astype('float32') 
	h = image.shape[0]
	w = image.shape[1]
	new_size = (int(h * scale), int(w * scale))
	return cv2.resize(image, new_size, interpolation = cv2.INTER_AREA)

def get_faces():
	data_images = fetch_olivetti_faces()
	data_target = data_images['target']
	data_images=data_images['images']
	return [data_images, data_target]

def read_faces_from_disk():
	data_faces = []
	data_target = []
	data_folder = os.path.dirname(os.path.abspath(__file__)) + "/faces/s"
	for i in range(1, 41):
		for j in range(1, 11):
			image = cv2.cvtColor(cv2.imread(data_folder + str(i) + "/" + str(j) + ".pgm"), cv2.COLOR_BGR2GRAY)
			data_faces.append(image/255)
			data_target.append(i)
	return [data_faces, data_target]

def data_for_example(data):
	return [data[0][62], data[0][60], data[0][48], data[0][44], data[0][157]]

def mesh_data(data):
	indexes = rnd.sample(range(0, len(data[0])), len(data[0]))
	return [data[0][index] for index in indexes], [data[1][index] for index in indexes]

def split_data(data, images_per_person_in_train=5, images_per_person_in_test=1):
	images_per_person = 10
	images_all = len(data[0])
	#if images_per_person_in_train > 8:
	#	images_per_person_in_train = 8
	#if images_per_person_in_test > 10 - images_per_person_in_train:
	#	images_per_person_in_test = 10 - images_per_person_in_train
	
	x_train, x_test, y_train, y_test = [], [], [], []

	for i in range(0, images_all, images_per_person):
		indices = list(range(i, i + images_per_person))
		indices_train = rnd.sample(indices, images_per_person_in_train)
		x_train.extend(data[0][index] for index in indices_train)
		y_train.extend(data[1][index] for index in indices_train)

		indices_test = rnd.sample(set(indices) - set(indices_train), images_per_person_in_test)
		x_test.extend(data[0][index] for index in indices_test)
		y_test.extend(data[1][index] for index in indices_test)
	
	return x_train, x_test, y_train, y_test

def choose_n_from_data(data, number):
	indexes = rnd.sample(range(0, len(data[0])), number)
	return [data[0][index] for index in indexes], [data[1][index] for index in indexes]

def create_feature(data, method, method_name, parameter):
	result = []
	for element in data:
		if method_name == 'get_histogram':
			result.append(method(element, parameter)[0])
		else:
			result.append(method(element, parameter))
	return result

def distance(el1, el2):
	return np.linalg.norm(np.array(el1) - np.array(el2))

def classifier(data, new_elements, method, method_name, parameter):
	if method_name not in ['get_histogram', 'get_dft', 'get_dct', 'get_gradient', 'get_scale']:
		return []
	featured_data = create_feature(data[0], method, method_name, parameter)
	featured_elements = create_feature(new_elements, method, method_name, parameter)
	result = []
	for element in featured_elements:
		min_el = [1000, -1]
		for i in range(len(featured_data)):
			dist = distance(element, featured_data[i])
			if dist < min_el[0]:
				min_el = [dist, i]
		if min_el[1] < 0:
			result.append(0)
		else:
			#print(min_el[1])
			result.append(data[1][min_el[1]])
	return result

def test_classifier(data, test_elements, method, method_name, parameter):
	if method_name not in ['get_histogram', 'get_dft', 'get_dct', 'get_gradient', 'get_scale']:
		return []
	answers = classifier(data, test_elements[0], method, method_name, parameter)
	correct_answers = 0
	for i in range(len(test_elements[1])):
		if answers[i] == test_elements[1][i]:
			correct_answers += 1
	return correct_answers/len(test_elements[1])


def teach_parameter(data, test_elements, method, method_name):
	if method_name not in ['get_histogram', 'get_dft', 'get_dct', 'get_gradient', 'get_scale']:
		return []
	image_size = min(data[0][0].shape)
	param = (0, 0, 0)
	if method_name == 'get_histogram':
		param = (10, 300, 3)
	if method_name == 'get_dft' or method_name == 'get_dct':
		param = (2, image_size, 1)
	if method_name == 'get_gradient':
		param = (2, int(data[0][0].shape[0]/2), 1)
	if method_name == 'get_scale':
		param = (0.05, 1, 0.01)
	
	best_param = param[0]
	classf = test_classifier(data, test_elements, method, method_name, best_param)
	stat = [[best_param], [classf]]

	for i in np.arange(param[0] + param[2], param[1], param[2]):
		new_classf = test_classifier(data, test_elements, method,method_name, i)
		stat[0].append(i)
		stat[1].append(new_classf)
		if new_classf > classf:
			classf = new_classf
			best_param = i
	
	return [best_param, classf], stat

def cross_validation(data, method, method_name, folds=3):
	#if folds < 3:
	#	folds = 3
	#per_fold = int(len(data[0])/folds)
	x_train = []
	x_test = []
	y_train = []
	y_test = []
	results = []
	for i in range(folds):
		print("fold " + str(i))
		for j in range(40):
			#print("j " + str(j) + " " + str(i+j*10))
			x_train.append(data[0][i+j*10])
			y_train.append(data[1][i+j*10])
			if folds == 5:
				x_test.append(data[0][i+j*10+folds])
				y_test.append(data[1][i+j*10+folds])
			else:
				if folds <5:
					x_test.append(data[0][i+j*10+folds])
					y_test.append(data[1][i+j*10+folds])
					if i == (folds-1):
						k = 10 - (folds*2)
						for kj in range(k):
							x_test.append(data[0][i+j*10+folds+kj+1])
							y_test.append(data[1][i+j*10+folds+kj+1])
				else: 
					k = abs(10 - (folds*2))
					if i <= (folds-(k+1)):
						x_test.append(data[0][i+j*10+folds])
						y_test.append(data[1][i+j*10+folds])
	results.append(teach_parameter([x_train, y_train], [x_test, y_test], method , method_name))
	
	#for step in range(0, folds):
	#	print("fold " + str(step))
		#if step == 0:
		#	x_train = data[0][per_fold:]
		#	x_test = data[0][:per_fold]
		#	y_train = data[1][per_fold:]
		#	y_test = data[1][:per_fold]
		#else:
		#	if step == folds - 1:
		#		x_train = data[0][:step*per_fold]
		#		x_test = data[0][step*per_fold:]
		#		y_train = data[1][:step*per_fold]
		#		y_test = data[1][step*per_fold:]
		#	else:
		#		x_train = data[0][:step*per_fold] + data[0][(step+1)*per_fold:]
		#		x_test = data[0][step*per_fold:(step+1)*per_fold]
		#		y_train = data[1][:step*per_fold] + data[1][(step+1)*per_fold:]
		#		y_test = data[1][step*per_fold:(step+1)*per_fold]
		#results.append(teach_parameter([x_train, y_train], [x_test, y_test], method , method_name))
	#print (results)
	res = results[0]
	for element in results[1:]:
		best = element[0]
		stat = element[1]
		res[0][0] += best[0]
		res[0][1] += best[1]
		for i in range(len(stat[1])):
			res[1][1][i] += stat[1][i]
	#-----------
	#res[0][0] /= folds
	#if method_name != 'get_scale':
	#	res[0][0] = int(res[0][0])
	#res[0][1] /= folds
	#for i in range(len(res[1][1])):
	#	res[1][1][i] /= folds
	print(res[1][0])
	return res