from PIL import Image, ImageTk
import tkinter as tki
from tkinter import filedialog
import threading
import datetime
import time
import imutils
from cv2 import cv2
import os
import sys
from face_classification import *
from functions_3 import *
from tkinter import messagebox as mb
from tkinter import Scrollbar
import io

class PhotoBoothApp:
	def __init__(self):
		self.outputPath = os.path.dirname(os.path.abspath(__file__)) + "/data"
		self.thread = None
		self.stopEvent = None
		self.root = tki.Tk()
		self.scroll_x = tki.Scrollbar(self.root, orient=tki.HORIZONTAL)
		self.scroll_y = tki.Scrollbar(self.root, orient=tki.VERTICAL)
		self.canvas = tki.Canvas(self.root, xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)
		self.root.title("Face classificator 4")
		self.root.protocol("WM_DELETE_WINDOW", self.onClose)
		self.root.bind('<Escape>', lambda e: self.root.quit())
		self.root.geometry("1150x650+50+50")
		self.root.resizable(False, False)
		self.canvas.grid(row=0, column=0)
		self.scroll_x.config(command=self.canvas.xview)
		self.scroll_y.config(command=self.canvas.yview)
		self.scroll_x.grid(row=1, column=0, sticky="we")
		self.scroll_y.grid(row=0, column=1, sticky="ns")
		self.frame_settings = tki.Frame(self.canvas, height=10080, width=2500)
		self.frame_settings.grid(column=0)
        
		self.canvas.config(width=1100, height=650)

		self.canvas.create_window((0, 0), window=self.frame_settings, anchor=tki.N + tki.W)
        
		self.root.bind("<Configure>", self.resize_s)
		self.root.update_idletasks()
		self.root.minsize(self.root.winfo_width(), self.root.winfo_height())

        #------------

		lbl1 = tki.Label(self.frame_settings, text="Количество эталонов")
		lbl1.grid(row=0, column=0, padx = 20, pady = 10)

		self.e1 = tki.Entry(self.frame_settings, width=20, bg='floral white')
		self.e1.grid(row=1, column=0, padx = 25, pady = 30, sticky='NE')

		btn1 = tki.Button(self.frame_settings, text="Далее",bg='floral white', command=self.get_folds)
		btn1.grid(row=2, column=0, sticky='N', padx = 25, pady = 8)

        #--------------

		lbl2 = tki.Label(self.frame_settings, text="Параметры")
		lbl2.grid(row=3, column=0, padx = 20, pady = 10)      

		lbl3 = tki.Label(self.frame_settings, text="Гистограмма: ")
		lbl3.grid(row=4, column=0, padx = 20, pady = 10) 

		self.e2 = tki.Entry(self.frame_settings, width=20, bg='floral white')
		self.e2.grid(row=4, column=1, padx = 25, pady = 30, sticky='NE')

		lbl4 = tki.Label(self.frame_settings, text="DFT: ")
		lbl4.grid(row=5, column=0, padx = 20, pady = 10) 

		self.e3 = tki.Entry(self.frame_settings, width=20, bg='floral white')
		self.e3.grid(row=5, column=1, padx = 25, pady = 30, sticky='NE')

		lbl5 = tki.Label(self.frame_settings, text="DCT: ")
		lbl5.grid(row=6, column=0, padx = 20, pady = 10) 

		self.e4 = tki.Entry(self.frame_settings, width=20, bg='floral white')
		self.e4.grid(row=6, column=1, padx = 25, pady = 30, sticky='NE')

		lbl6 = tki.Label(self.frame_settings, text="Градиент: ")
		lbl6.grid(row=7, column=0, padx = 20, pady = 10) 

		self.e5 = tki.Entry(self.frame_settings, width=20, bg='floral white')
		self.e5.grid(row=7, column=1, padx = 25, pady = 30, sticky='NE')

		lbl7 = tki.Label(self.frame_settings, text="Scale: ")
		lbl7.grid(row=8, column=0, padx = 20, pady = 10) 

		self.e6 = tki.Entry(self.frame_settings, width=20, bg='floral white')
		self.e6.grid(row=8, column=1, padx = 25, pady = 30, sticky='NE')

		btn2 = tki.Button(self.frame_settings, text="Далее",bg='floral white', command=self.get_parameters)
		btn2.grid(row=9, column=1, sticky='N', padx = 25, pady = 8)

        #---------------

		lbl8 = tki.Label(self.frame_settings, text=" ")
		lbl8.grid(row=10, column=0, padx = 20, pady = 10) 

		self.e7 = tki.Entry(self.frame_settings, width=20, bg='floral white')
		self.e7.grid(row=11, column=0, padx = 25, pady = 30, sticky='NE')

		btn3 = tki.Button(self.frame_settings, text="Старт",bg='floral white', command=self.get_pic_number_and_start)
		btn3.grid(row=10, column=1, sticky='N', padx = 25, pady = 8)

        #-------------
        
		self.data = get_faces()
		self.parameters = []
		self.res = []
		self.folds = 0
		self.pic_number = 0
		self.res_number = 0
		self.img_massiv = []
		self.print_massiv_number = 0
		self.pic_method_r = []
		self.pic_method = []
		self.res_method = []
		self.res_method_r = []
		self.pic = []
		self.pic_method = []
		self.label_diap = tki.Label(self.frame_settings, text="")
		self.label_diap.grid(row=0, column=2, sticky='NW')

		j=0
		for i in range(40):
			self.pic.append(tki.Label(self.frame_settings))
			self.pic[i].grid(row = 1+j, column = 2, sticky='N', padx=10, pady=2)
			j=j+4
		for i in range(200):
			self.pic_method.append(tki.Label(self.frame_settings))
		k=0
		l=0
		for i in range(40):
			for j in range(5):
				self.pic_method[k].grid(row = 2+l, column = j+2, sticky='N', padx=10, pady=2)
				k=k+1
			l=l+4
		self.img_results =[]
		for i in range(200):
			self.img_results.append(tki.Label(self.frame_settings))
		k=0
		l=0
		for i in range(40):
			for j in range(5):
				self.img_results[k].grid(row = 3+l, column = j+2, sticky='N', padx=10, pady=2)
				k=k+1
			l=l+4

		for i in range(200):
			self.res_method.append(tki.Label(self.frame_settings))
		k=0
		l=0
		for i in range(40):
			for j in range(5):
				self.res_method[k].grid(row = 4+l, column = j+2, sticky='N', padx=10, pady=2)
				k=k+1
			l=l+4

		#lbl9 = tki.Label(self.frame_settings, text="Результат: ")
		#lbl9.grid(row=3, column=2, padx = 20, pady = 10) 

		#self.pic_res = tki.Label(self.frame_settings)
		#self.pic_res.grid(row = 4, column = 4, sticky='N', padx=10, pady=2)

		#lbl10 = tki.Label(self.frame_settings, text="Человек под номером")
		#lbl10.grid(row=4, column=2, padx = 25, sticky='N')
		#self.label_pic_res = tki.Label(self.frame_settings, text="")
		#self.label_pic_res.grid(row=4, column=3, sticky='NW')

	def get_folds(self):
		f1 = float(self.e1.get())
		self.folds = int(f1)
		self.label_diap.configure(text = "Тестовое изображение можно выбрать из [" + str(0) + "," + str(399 - 40*(self.folds))+"]")
		self.print_massiv_number = 399 - 40*(self.folds)
		print("Тестовое изображение можно выбрать из [" + str(0) + "," + str(399 - 40*(self.folds))+"]")

	def get_parameters(self):
		p1 = float(self.e2.get())      
		p2 = float(self.e3.get()) 
		p3 = float(self.e4.get())  
		p4 = float(self.e5.get())
		p5 = float(self.e6.get())
		self.parameters = []
		self.parameters.append(int(p1)) 
		self.parameters.append(int(p2))
		self.parameters.append(int(p3)) 
		self.parameters.append(int(p4))   
		self.parameters.append(p5)  
		#print(self.parameters)

	def get_pic_number_and_start(self):
		k=0
		l=0
		m=0
		for j in range(self.print_massiv_number + 1):
		#for j in range(1):
			self.pic_number = j
			self.res = start_classification_hard(self.folds, self.pic_number, self.parameters)
			self.img_massiv = self.res[0]		
			self.res_number = self.res[1]
			pic_index = self.pic_number + self.folds * ((self.pic_number // (10 - self.folds))+1)
			#img_res = self.res_number*10
			image = Image.fromarray(self.data[0][pic_index]*255)
			image = ImageTk.PhotoImage(image)
			self.pic[j].configure(image=image)
			self.pic[j].image = image
			self.pic_method_r = get_method_pic(self.parameters, self.data[0][pic_index])

			method_name = ['get_histogram', 'get_dft', 'get_dct', 'get_gradient', 'get_scale']
			#method = [get_histogram, get_dft_here, get_dct, get_gradient_el, get_scale_here]
			for i in range(5):
				if method_name[i] == 'get_histogram':
					hist, bins = self.pic_method_r[i]
					hist = np.insert(hist, 0, 0.0)
					fig = plt.figure(figsize=(1.1,1.1))
					ax = fig.add_subplot(111)
					ax.plot(bins, hist)
					plt.xticks(color='w')
					plt.yticks(color='w')
					buf = io.BytesIO()
					fig.savefig(buf)
					buf.seek(0)
					image = Image.open(buf)
					image = ImageTk.PhotoImage(image)
					self.pic_method[k].configure(image=image)
					self.pic_method[k].image = image
					k=k+1
				if method_name[i] == 'get_dft':
					fig = plt.figure(figsize=(1.1,1.1))
					ax = fig.add_subplot(111)
					ax.pcolormesh(range(self.pic_method_r[i].shape[0]),
										range(self.pic_method_r[i].shape[0]),
										np.flip(self.pic_method_r[i], 0), cmap="Greys")
					plt.xticks(color='w')
					plt.yticks(color='w')
					buf = io.BytesIO()
					fig.savefig(buf)
					buf.seek(0)
					image = Image.open(buf)
					image = ImageTk.PhotoImage(image)
					self.pic_method[k].configure(image=image)
					self.pic_method[k].image = image
					k=k+1
				if method_name[i] == 'get_dct':
					fig = plt.figure(figsize=(1.1,1.1))
					ax = fig.add_subplot(111)
					ax.pcolormesh(range(self.pic_method_r[i].shape[0]),
										range(self.pic_method_r[i].shape[0]),
										np.flip(self.pic_method_r[i], 0), cmap="Greys")
					plt.xticks(color='w')
					plt.yticks(color='w')
					buf = io.BytesIO()
					fig.savefig(buf)
					buf.seek(0)
					image = Image.open(buf)
					image = ImageTk.PhotoImage(image)
					self.pic_method[k].configure(image=image)
					self.pic_method[k].image = image
					k=k+1
				if method_name[i] == 'get_scale':
					image = Image.fromarray(cv2.resize(self.pic_method_r[i]*255, self.pic_method_r[i].shape, interpolation = cv2.INTER_AREA))
					image = ImageTk.PhotoImage(image)
					self.pic_method[k].configure(image=image)
					self.pic_method[k].image = image
					k=k+1
				if method_name[i] == 'get_gradient':
					#image_size = self.data[0][0].shape[0]
					fig = plt.figure(figsize=(1.1,1.1))
					ax = fig.add_subplot(111)
					ax.plot(range(0, len(self.pic_method_r[i])), self.pic_method_r[i])
					plt.xticks(color='w')
					plt.yticks(color='w')
					buf = io.BytesIO()
					fig.savefig(buf)
					buf.seek(0)
					image = Image.open(buf)
					image = ImageTk.PhotoImage(image)
					self.pic_method[k].configure(image=image)
					self.pic_method[k].image = image
					k=k+1
			for i in range(len(self.img_massiv)):
				image = Image.fromarray(self.data[0][self.img_massiv[i]]*255)
				image = ImageTk.PhotoImage(image)
				self.img_results[l].configure(image=image)
				self.img_results[l].image = image
				l=l+1
				self.res_method_r = get_method_pic(self.parameters, self.data[0][self.img_massiv[i]])
				if method_name[i] == 'get_histogram':
					hist, bins = self.res_method_r[i]
					hist = np.insert(hist, 0, 0.0)
					fig = plt.figure(figsize=(1.1,1.1))
					ax = fig.add_subplot(111)
					ax.plot(bins, hist)
					plt.xticks(color='w')
					plt.yticks(color='w')
					buf = io.BytesIO()
					fig.savefig(buf)
					buf.seek(0)
					image = Image.open(buf)
					image = ImageTk.PhotoImage(image)
					self.res_method[m].configure(image=image)
					self.res_method[m].image = image
					m=m+1
				if method_name[i] == 'get_dft':
					fig = plt.figure(figsize=(1.1,1.1))
					ax = fig.add_subplot(111)
					ax.pcolormesh(range(self.res_method_r[i].shape[0]),
										range(self.res_method_r[i].shape[0]),
										np.flip(self.res_method_r[i], 0), cmap="Greys")
					plt.xticks(color='w')
					plt.yticks(color='w')
					buf = io.BytesIO()
					fig.savefig(buf)
					buf.seek(0)
					image = Image.open(buf)
					image = ImageTk.PhotoImage(image)
					self.res_method[m].configure(image=image)
					self.res_method[m].image = image
					m=m+1
				if method_name[i] == 'get_dct':
					fig = plt.figure(figsize=(1.1,1.1))
					ax = fig.add_subplot(111)
					ax.pcolormesh(range(self.res_method_r[i].shape[0]),
										range(self.res_method_r[i].shape[0]),
										np.flip(self.res_method_r[i], 0), cmap="Greys")
					plt.xticks(color='w')
					plt.yticks(color='w')
					buf = io.BytesIO()
					fig.savefig(buf)
					buf.seek(0)
					image = Image.open(buf)
					image = ImageTk.PhotoImage(image)
					self.res_method[m].configure(image=image)
					self.res_method[m].image = image
					m=m+1
				if method_name[i] == 'get_scale':
					image = Image.fromarray(cv2.resize(self.res_method_r[i]*255, self.res_method_r[i].shape, interpolation = cv2.INTER_AREA))
					image = ImageTk.PhotoImage(image)
					self.res_method[m].configure(image=image)
					self.res_method[m].image = image
					m=m+1
				if method_name[i] == 'get_gradient':
					#image_size = self.data[0][0].shape[0]
					fig = plt.figure(figsize=(1.1,1.1))
					ax = fig.add_subplot(111)
					ax.plot(range(0, len(self.res_method_r[i])), self.res_method_r[i])
					plt.xticks(color='w')
					plt.yticks(color='w')
					buf = io.BytesIO()
					fig.savefig(buf)
					buf.seek(0)
					image = Image.open(buf)
					image = ImageTk.PhotoImage(image)
					self.res_method[m].configure(image=image)
					self.res_method[m].image = image
					m=m+1


	def resize_s(self, event):
		region = self.canvas.bbox(tki.ALL)
		self.canvas.configure(scrollregion=region)

	def onClose(self):
		print("[INFO] closing...")
		self.root.quit()

pba = PhotoBoothApp()
pba.root.mainloop()
