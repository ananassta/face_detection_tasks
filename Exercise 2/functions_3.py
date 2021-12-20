
# классификатор голосованием

from sklearn.ensemble import VotingClassifier
from face_classification import *
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_iris

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split


def split_train_test(data, target, images_per_person_in_train):
	images_per_person = 10
	images_all = len(data)
	images_per_person_in_test = images_per_person - images_per_person_in_train
	#if images_per_person_in_train > 8:
	#	images_per_person_in_train = 8
	#if images_per_person_in_test > 10 - images_per_person_in_train:
	#	images_per_person_in_test = 10 - images_per_person_in_train
	
	x_train, x_test, y_train, y_test = [], [], [], []

	for i in range(0, images_all, images_per_person):
		indices = list(range(i, i + images_per_person))
		indices_train = rnd.sample(indices, images_per_person_in_train)
		x_train.extend(data[index] for index in indices_train)
		y_train.extend(target[index] for index in indices_train)

		indices_test = rnd.sample(set(indices) - set(indices_train), images_per_person_in_test)
		x_test.extend(data[index] for index in indices_test)
		y_test.extend(target[index] for index in indices_test)
	
	return x_train, x_test, y_train, y_test

def split_train_test_mine(data, target, folds):
	x_train = []
	x_test = []
	y_train = []
	y_test = []
	results = []
	for i in range(folds):
		#print("fold " + str(i))
		for j in range(40):
			#print("j " + str(j) + " " + str(i+j*10))
			x_train.append(data[i+j*10])
			y_train.append(target[i+j*10])
			if folds == 5:
				x_test.append(data[i+j*10+folds])
				y_test.append(target[i+j*10+folds])
			else:
				if folds <5:
					x_test.append(data[i+j*10+folds])
					y_test.append(target[i+j*10+folds])
					if i == (folds-1):
						k = 10 - (folds*2)
						for kj in range(k):
							x_test.append(data[i+j*10+folds+kj+1])
							y_test.append(target[i+j*10+folds+kj+1])
				else: 
					k = abs(10 - (folds*2))
					if i <= (folds-(k+1)):
						x_test.append(data[i+j*10+folds])
						y_test.append(target[i+j*10+folds])
	results.append(x_train)
	results.append(x_test)
	results.append(y_train)
	results.append(y_test)
	return results

def get_scale_here(image, scale = 0.35):
	image = image.astype('float32') 
	h = image.shape[0]
	w = image.shape[1]
	new_size = (int(h * scale), int(w * scale))
	return cv2.resize(image, new_size, interpolation = cv2.INTER_AREA)

def estimator_hard(X_train, x_test_i,folds,parameter):
    methods_names = ['get_histogram', 'get_dft', 'get_dct', 'get_gradient', 'get_scale']
    methods = [get_histogram, get_dft_here, get_dct, get_gradient_el, get_scale_here]
    #parameter = [43, 13, 8, 3, 0.3]
    test_results = []
    for i in range(5):
        m = methods[i]
        m_name = methods_names[i]
        test_results.append(test_classifier_H(X_train, x_test_i, m, m_name, parameter[i])) #получили i - номер тестового изображения, что ближе всех
    #print (test_results)
    for i in range(5):
        v = test_results[i]
        a = int(''.join(str(j) for j in v))
        b = (a//folds)*(10 - folds) + a
        test_results[i] = b
    print (test_results)
    return test_results

def get_dft_here(image, mat_side = 13):
    f = np.fft.fft2(image)
    f = f[0:mat_side, 0:mat_side]
    return np.abs(f)

def test_classifier_H(data, test_elements, method, method_name, parameter):
	#if method_name not in ['get_histogram', 'get_dft', 'get_dct', 'get_gradient', 'get_scale']:
	#	return []
	#print (test_elements[0])
	#print("------")
	answers = classifier_H(data, test_elements[0], method, method_name, parameter)
	#correct_answers = answers[0]
	#for i in range(len(answers))):
	#	if answers[i] == test_elements[1][i]:
	#		correct_answers += 1
	return answers

def classifier_H(data, new_elements, method, method_name, parameter):
	#if method_name not in ['get_histogram', 'get_dft', 'get_dct', 'get_gradient', 'get_scale']:
	#	return []
	featured_data = create_feature_H(data[0], method, method_name, parameter)
	#print (new_elements)
	featured_elements = create_feature_H_el(new_elements, method, method_name, parameter)
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
			result.append(min_el[1])
	return result

def create_feature_H(data, method, method_name, parameter):
	result = []
	for i in range(len(data)):
		if method_name == 'get_histogram':
			result.append(method(data[i], parameter)[0])
		else:
			result.append(method(data[i], parameter))
	return result

def create_feature_H_el(data, method, method_name, parameter):
	result = []
	if method_name == 'get_histogram':
		result.append(method(data, parameter)[0])
	else:
		result.append(method(data, parameter))
	return result

def get_gradient_el(image, n = 2):
    n=n-1
    shape = image.shape[0]
    #print(shape)
    i, l = 0, 0
    r = n
    result = []

    while r <= shape:
        window = image[l:r][:]
        result.append(np.sum(window))
        i += 1
        l = i * n
        r = (i + 1) * n
    result = np.array(result)
    return result

def start_classification_hard(folds, test_image_number, parameters_m):
    # загрузка набора данных 
    data = get_faces()

    X = data[0]
    #print(X[0])
    #print("---------")
    #X = X.reshape(X.shape[0], -1)
    #print(X)
    Y = data[1]
    #Y = Y.reshape(Y.shape[0], -1)
    
    # train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)
    X_train, X_test, y_train, y_test = split_train_test(X,Y,folds)
    train = [X_train,y_train]
    #print (X_train[279])
    x_test_i = X_test[test_image_number]
    y_test_i = y_test[test_image_number]
    test = [x_test_i,y_test_i]
    #print (test[0])
    #f = get_dft_here(test[0], 13)
    #print (f)
    #print (10//2)
    # группа / ансамбль моделей

    #задать параметры и объемы обучающей выборки
    #выбрать картинку из тестовой выборки (вызвать ее на экран)
    #для каждого из методов вызвать тест классифаер и найти голоса для картинки

    #получили массив голосов от каждого метода (жесткое голосование)
    test_results = estimator_hard(train, test, folds, parameters_m)

    pic = []
    face_number = []
    for i in range(5):
        pic.append(test_results[i]%10) #для вывода картинки на экран
        face_number.append(test_results[i]//10)
    votes = []
    for i in range(40):
        votes.append(0)
    #print(face_number)
    for i in range(5):
        votes[face_number[i]] = votes[face_number[i]]+1
    #print (votes)
    max_i = 0
    index = 0
    for i in range(40):
        if votes[i]> max_i:
            max_i = votes[i]
            index = i
    print ("Hard voting shows - " + str(index))
    print ("рассчет от 0")
    return test_results, index

def get_method_pic(parameters, pic):
	methods = [get_histogram, get_dft_here, get_dct, get_gradient_el, get_scale_here]
	res = []
	j=0
	for i in methods:
		res.append(i(pic, parameters[j]))
		j=j+1
	return res

#test_results = []
#index = 0
#test_results, index = start_classification_hard(7, 1, [43, 13,8,3,0.3])

#estimator = []
#estimator.append(('methods', estimator_hard(X_train, x_test_i)))

#estimator.append(('LR', LogisticRegression(solver ='lbfgs', multi_class ='multinomial', max_iter = 200)))

#estimator.append(('SVC', SVC(gamma ='auto', probability = True)))

#estimator.append(('DTC', DecisionTreeClassifier()))

  
# Голосующий классификатор с жестким голосованием

#vot_hard = VotingClassifier(estimators = estimator, voting ='hard')
#print (X_train)
#print("-------------")
#X_train = X_train.reshape(X_train.shape[0], -1)
#print(X_train)

#vot_hard.fit(X_train, y_train)

#y_pred = vot_hard.predict(x_test_i)

  
# использование метрики precision_score для прогнозирования точности

#score = accuracy_score(y_test_i, y_pred)

#print("Hard Voting Score % d" % score)

  
# Классификатор голосования с мягким голосованием

#vot_soft = VotingClassifier(estimators = estimator, voting ='soft')

#vot_soft.fit(X_train, y_train)

#y_pred = vot_soft.predict(X_test)

  
# используя precision_score

#score = accuracy_score(y_test, y_pred)

#print("Soft Voting Score % d" % score)