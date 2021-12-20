#подключаем базу данных лиц
from sklearn.datasets import fetch_olivetti_faces
data_images = fetch_olivetti_faces()
data_images=data_images['images']

#выводим первую картинку
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
#fx, ax = plt.subplots()
#ax.imshow(data_images[0])
#plt.imshow(data_images[0])
#plt.show()

cv2_imshow(data_images[0]*255)
cv2.waitKey(0)
cv2.destroyAllWindows()

#реализуем детектор лиц на основе метода Template Matching
import numpy as np
import cv2

templates = data_images[::10]*255
cv2.imwrite('template.jpg', templates[0])
template = cv2.imread('template.jpg',0)
template = cv2.resize(template, (500, 500), interpolation = cv2.INTER_LINEAR)

img = cv2.imread('lor.jpeg',0)
#img = cv2.resize(img, (1500,1000), interpolation = cv2.INTER_LINEAR)

#img = cv2.imread('lor.jpeg',0)
img2 = img.copy()
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()
    
    #с плохим освещением

template = cv2.imread('0.jpg',0)
template = cv2.resize(template, (500, 500), interpolation = cv2.INTER_LINEAR)

img = cv2.imread('frodo_bad_light.jpg',0)
#img = cv2.imread('lor.jpeg',0)
img2 = img.copy()
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()
    
    #с неравномерным освещением

template = cv2.imread('0.jpg',0)
template = cv2.resize(template, (500, 500), interpolation = cv2.INTER_LINEAR)

img = cv2.imread('frodo_nerevnomernyi_light.png',0)
#img = cv2.imread('lor.jpeg',0)
img2 = img.copy()
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()
    
    #отдаление от камеры

template = cv2.imread('template.jpg',0)
template = cv2.resize(template, (500, 500), interpolation = cv2.INTER_LINEAR)

img = cv2.imread('frodo_daleko.png',0)
img = cv2.resize(img, (2410, 1500), interpolation = cv2.INTER_LINEAR)
#img = cv2.imread('lor.jpeg',0)
img2 = img.copy()
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()
    
    #скрыта часть лица 

template = cv2.imread('template.jpg',0)
template = cv2.resize(template, (500, 500), interpolation = cv2.INTER_LINEAR)

img = cv2.imread('frodo_skrito.jpg',0)
img = cv2.resize(img, (2410, 1500), interpolation = cv2.INTER_LINEAR)
#img = cv2.imread('lor.jpeg',0)
img2 = img.copy()
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()
    
    #поворот головы

template = cv2.imread('template.jpg',0)
template = cv2.resize(template, (500, 500), interpolation = cv2.INTER_LINEAR)

img = cv2.imread('povorot.jpg',0)
img = cv2.resize(img, (2410, 1500), interpolation = cv2.INTER_LINEAR)
#img = cv2.imread('lor.jpeg',0)
img2 = img.copy()
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()
    
    #наклон головы

template = cv2.imread('template.jpg',0)
template = cv2.resize(template, (1000, 1000), interpolation = cv2.INTER_LINEAR)

img = cv2.imread('Frodo_naklon.jpg',0)
img = cv2.resize(img, (2410, 1500), interpolation = cv2.INTER_LINEAR)
#img = cv2.imread('lor.jpeg',0)
img2 = img.copy()
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()
    
    #детекторы с различными шаблонами (фрагментами лица)

#сама картинка
#img = cv2.imread('lor.jpeg',0)
img = cv2.imread('frodo.jpg',0)
img2 = img.copy()

# All the 6 methods for comparison in a list
methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']
for i in range(3):
  if i==0:
    #вертикальная половина лица
    plt.suptitle('vertical')
    template = cv2.imread('0_1.jpg',0)
  elif i==1:
    #глаз
    plt.suptitle('eye')
    template = cv2.imread('0_2.jpg',0)
  else:
    #горизонтальная половина лица 
    plt.suptitle('horizontal')
    template = cv2.imread('0_3.jpg',0)
  template = cv2.resize(template, (500, 500), interpolation = cv2.INTER_LINEAR)
  w, h = template.shape[::-1]
  for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res,cmap = 'gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img,cmap = 'gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()
    
#import numpy as np
#import cv2
#from google.colab.patches import cv2_imshow

#задание 3

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

img = cv2.imread('skrito.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    #eyes = eye_cascade.detectMultiScale(roi_gray)
    #for (ex,ey,ew,eh) in eyes:
    #    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2_imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#задание 4

# import the necessary packages
from imutils import face_utils
import imutils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from math import sqrt

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True,
#	help="file")
#ap.add_argument("-i", "--image", required=True,
#	help="frodo.jpg")
#args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('file')

# load the input image, resize it, and convert it to grayscale
image = cv2.imread('Bvx2D-SpfWw.jpg')
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
  shape = predictor(gray, rect)
  shape = face_utils.shape_to_np(shape)
  # convert dlib's rectangle to a OpenCV-style bounding box
  # [i.e., (x, y, w, h)], then draw the face bounding box
  (x, y, w, h) = face_utils.rect_to_bb(rect)
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
  # show the face number
  #cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
  #	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
  # loop over the (x, y)-coordinates for the facial landmarks
  # and draw them on the image
  (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
  (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
  leftEye = shape[lStart:lEnd]
  rightEye = shape[rStart:rEnd]
  (xl1,yl1)=shape[37]
  (xl2,yl2)=shape[38]
  (xl3,yl3)=shape[40]
  (xl4,yl4)=shape[41]
  cl1 = int(sqrt((xl2-xl1)**2 + (yl2-yl1)**2)/2)
  cl2 = int(sqrt((xl4-xl3)**2 + (yl4-yl3)**2)/2)
  l_start = (xl1+cl1,yl1+cl1)
  l_end = (xl4+cl2,yl4+cl2)
  #cv2.line(image,l_start,l_end,(0, 240, 240),3)
  #print (l_start)
  #k = int((yl1+cl1 - yl4+cl2) / (xl1+cl1 - xl4+cl2))
  #b = yl4+cl2 - k * (xl4+cl2)
  #print(k*203+b)
  #        writeln('y = ',k:0:2,'x + ',b:0:2);
  #cv2.line(image, (203,k*203+b), (250, k*250+b), (0,0,0), 2)

  (xf,yf) = shape[8]
  (xs,ys) = shape[27]
  cv2.line(image,(xf,yf),(xs,ys),(0, 240, 240),3)

  #print (xs,ys)
  xl1r = xs - (xl1+cl1)
  xl2 = xf - xl1r
  cv2.line(image,(xl2,yf),(xl1+cl1,ys),(255, 0, 0),3)

  (xr1,yr1)=shape[43]
  (xr2,yr2)=shape[44]
  (xr3,yr3)=shape[46]
  (xr4,yr4)=shape[47]
  cr1 = int(sqrt((xr2-xr1)**2 + (yr2-yr1)**2)/2)
  cr2 = int(sqrt((xr4-xr3)**2 + (yr4-yr3)**2)/2)
  r_start = (xr1+cr1,yr1+cr1)
  r_end = (xr4+cr2,yr4+cr2)
  xr1r = (xr1+cr1) - xs
  xr2 = xf + xr1r
  cv2.line(image,(xr2,yf),(xr1+cr1,ys),(255, 0, 0),3)

  (xrf,yrf) = shape[24]
  (xrs,yrs) = shape[10]
  #cv2.line(image,(xrf,yrf),(xrs,yrs),(255, 0, 0),3)
  (xlf,ylf) = shape[48]
  (xls,yls) = shape[20]
  #cv2.line(image,(xlf,ylf),(xls,yls),(255, 0, 0),3)
  #for (x, y) in leftEye:
   # cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
  #for (x, y) in rightEye:
   # cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
  #for (x, y) in shape:
    #print ((x, y))
   # cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
cv2_imshow(image)
cv2.waitKey(0)
