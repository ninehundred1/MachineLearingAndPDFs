__author__ = 'Meyer'

"""
GUI that deletes from a multi page PDF documents unwanted pages (such as empty scans)
using machine learning.
A sample PDF can be extracted to single page jpgs using the 'export one PDF ...' function,
then after the neural network can be trained by loading all unwanted jpgs first to train,
followed by exampled of jpgs to keep. 
The then trained network can be used to automatically delete unwanted pdfs from other
documents.
"""

import os
import Tkinter
from tkFileDialog import askopenfilename,askdirectory
import copy, sys
from PythonMagick import Image
from PIL import Image, ImageTk, ImageDraw
from StringIO import StringIO
import numpy as np
import matplotlib.pyplot as plt


class Delete_PDF():
	
	def __init__(self, parent_gui):
		self.root = parent_gui
		self.create_GUI()
		self.num_files = 0
		self.num_files_counter = 0
		self.file_list = []
		self.log = []
		self.max_pixel_val = []
		self.features_full = np.empty([0,0])
		self.features_empty = np.empty([0,0])
		self.new_image_size = (70, 40)

		
	def create_GUI(self):
		self.root.wm_title("DeletePDF")
		instruction_field = Tkinter.Label(self.root, 
			text="Choose single .pdf file or a folder containg PDFs to upload.")
		instruction_field.pack(ipadx=50)
		self.choose_file_button = Tkinter.Button(self.root, 
			text="choose single .pdf file", fg="blue", command=self.on_load_single_pdf)
		self.choose_file_button.pack(ipadx=50)
		self.choose_file_button = Tkinter.Button(self.root, 
			text="export one PDF to single jpgs for training", fg="green", command=self.on_load_export_jpg)
		self.choose_file_button.pack(ipadx=50)
		self.choose_folder_button = Tkinter.Button(self.root, 
			text="choose folder with pdfs", fg="red", command=self.on_load_folder)
		self.choose_folder_button.pack(ipadx=50)
		self.choose_folder_button = Tkinter.Button(self.root, 
			text="Train with new empty and full jpgs", fg="black", bg="green", command=self.on_do_train)
		self.choose_folder_button.pack(ipadx=50)
		self.choose_folder_button = Tkinter.Button(self.root, 
			text="Load Training Classifiers", fg="white", bg="blue", command=self.load_training)
		self.choose_folder_button.pack(ipadx=50)
		instruction_field2 = Tkinter.Label(self.root, 
			text="The PDF without empty pages as well as deleted pages will be saved in same directory.")
		instruction_field2.pack(ipadx=50)
		self.file_text = Tkinter.Label(self.root,  font=("Helvetica", 9), text=" - ")
        	self.file_text.pack()
        	self.status_text = Tkinter.Label(self.root,  font=("Helvetica", 11), text=" Please load file(s)")
        	self.status_text.pack()
        	self.log_text = Tkinter.Text(self.root, width = 80, height = 10, takefocus=0)
        	self.scrollbar = Tkinter.Scrollbar(self.root)
        	self.scrollbar.pack(side = Tkinter.RIGHT, fill=Tkinter.Y )
        	self.scrollbar.config(command=self.log_text.yview)
        	self.log_text.config(yscrollcommand=self.scrollbar.set)
        	self.log_text.pack()
		self.image_panel = Tkinter.Label(self.root)


	def load_training(self):
		pass
	
	def on_do_train(self):
		self.log_text.insert(Tkinter.INSERT, 
			'\n  ****TRAINING****\n' )
		self.log_text.insert(Tkinter.INSERT, 'Reducing all image sizes to ' + str(self.new_image_size))
		self.log_text.see(Tkinter.END) 
		process_directory = askdirectory(parent=root,
			title='Choose folder with FULL pdf image files only',)
		if process_directory:
			self.log_text.insert(Tkinter.INSERT, '\nLoad Path Full images:\n' )
			self.log_text.insert(Tkinter.INSERT, process_directory )
			self.log_text.see(Tkinter.END) 
			self.features_full = self.process_directory(process_directory)

			process_directory = askdirectory(initialdir = process_directory, parent=root,
				title='Choose folder with EMPTY pdf image files only',)
			self.log_text.insert(Tkinter.INSERT, '\nLoad Path Empty images:\n' )
			self.log_text.insert(Tkinter.INSERT, process_directory )
			self.log_text.see(Tkinter.END) 
			self.features_empty = self.process_directory(process_directory)
		else:
			self.status_text.configure(text="File path invalid")

		


	def process_directory(self, directory):
		'''Given a directory path the function iterates through all images
		in folder and passes each file name into process_image to extract
		that image features. The returned feature 1D array get appended 
		to a tensor array with [n_image, image_pixel] dimensions. Then the
		max 8 bit value of the Full set gets used to normalize both, full 
		and empty tensors to a float between 0-1.
		With the empty set being scanner noise, the lower values might help
		to classify.

		Args:
		  directory (str): directory path of images.

		Returns:
		  tensor array (array of floats): feature array if successful, otherwise
		  return None.
		'''
		feature_array = []
		successes = 0
		#to count files in multiple folders use an extra counter
		counter = 0
		for root, _, files in os.walk(directory):
			for i, file_name in enumerate(files):
				text_string = 'current image: %s' %(i+1)
				self.file_text.configure(text=text_string)
				file_path = os.path.join(root, file_name)
				img_feature = self.process_image_file(file_path)
				#only append if feature extracted
				if img_feature is not None:
					feature_array.append(img_feature)
					successes = successes + 1
				counter = counter + 1
		feature_array = np.vstack(feature_array)
		#get max pixel value of full set before normalization
		if not self.max_pixel_val:
			self.max_pixel_val = feature_array.max()
		#normalize to 0-1
		feature_array = feature_array/self.max_pixel_val
		if self.features_full.size is 0:
			current_set = 'FULL images'
		else:
			current_set = 'EMPTY images'
		text_string = "\nConverted %s of %s to features" \
			% ((str(successes) + '/' + str(counter)), current_set)
		self.log_text.insert(Tkinter.INSERT, text_string)
		text_string = "\nPreview:" 
		self.log_text.insert(Tkinter.INSERT, text_string)
		prev_array = np.array(feature_array[0:7,0:7])
		preview = np.array(["%.2f" % w for w in prev_array.reshape(prev_array.size)])
		preview = preview.reshape(prev_array.shape)
		text_string = "\n%s" % preview
		self.log_text.insert(Tkinter.INSERT, text_string)
		self.log_text.insert(Tkinter.INSERT, '\nMean value: ' + str(np.mean(feature_array)))
		self.log_text.insert(Tkinter.INSERT, '\nMin value: ' + str(np.amin(feature_array)))
		self.log_text.insert(Tkinter.INSERT, '\nMax value: ' + str(np.amax(feature_array)))
		self.log_text.see(Tkinter.END) 
		return feature_array

	def process_image_file(self, image_path):
		'''Given an image path it reduces each image in size to new_image_size
		and then transforms it to a 1D list and to an nparray

		Args:
		  image_path (str): path to current image.

		Returns:
		  list of ints (8bit): feature vector if success, None if not.
		'''
		try:
			img = Image.open(image_path)
			img = img.resize(self.new_image_size)
			img = list(img.getdata())
			img = np.array(img)
		except IOError:
			return None
		return img

	
	def on_load_single_pdf(self):
		self.num_pages_counter = 0
		file = askopenfilename(filetypes=[("pdf file","*.pdf")], parent=root,
			title='Choose .pdf file to load')
		if file:
			self.num_files = 1
			self.status_text.configure(text="Loaded single PDF")
			self.file_name = os.path.splitext(file)[0]
			name_ext=os.path.basename(file)
			self.name_only = os.path.splitext(name_ext)[0]
			self.status_text.configure(text="Extracting Images..")
			#self.export_to_jpg(file, name_only)
			#self.delete2(file)
			self.hand(file)
			self.status_text.configure(text="All Done")
		else:
			self.status_text.configure(text="File path invalid")


	def on_load_folder(self):
		file = askopenfilename(filetypes=[("pdf file","*.pdf")], parent=root,
			title='Choose .pdf file to load')
		if file:
			self.status_text.configure(text="Loaded single PDF")
			self.file_name = os.path.splitext(file)[0]
			name_ext=os.path.basename(file)
			name_only = os.path.splitext(name_ext)[0]
			self.log_text.insert(Tkinter.INSERT, '\nLoad Path:' + name_only)
									
		else:
			self.status_text.configure(text="File path invalid")

	def on_load_export_jpg(self):
		'''Asks for image path which is then passed into the export_to_jpg
		function which converts each page (the image on the page) of PDF into an separate jpg. 
		Args:
		  none

		Returns:
		  none.
		'''
		self.log_text.insert(Tkinter.INSERT, 
			'\n  ****EXTRACT JPGS****' )
		self.log_text.see(Tkinter.END) 
		file = askopenfilename( parent=root,
			title='Choose .pdf file to load')
		if file:
			self.file_name = os.path.splitext(file)[0]
			name_ext=os.path.basename(file)
			name_only = os.path.splitext(name_ext)[0]
			directory = os.path.split(file)[0]
			new_directory = directory+'//ExtractedJPGs//'
			if not os.path.exists(new_directory):
				os.makedirs(new_directory)

			self.log_text.insert(Tkinter.INSERT, '\nExtracing JPGS from %s'
			 % self.file_name)
			self.log_text.insert(Tkinter.INSERT, '\nSaving JPGS to %s'
			 % new_directory)
			self.status_text.configure(text="Extracting Images..")
			num_extracted = self.export_to_jpg(file, name_only, new_directory)
			report_text = "\nAll Done, extracted %s images" % str(num_extracted)
			self.status_text.configure(text=report_text)
			self.log_text.insert(Tkinter.INSERT, report_text)
			self.log_text.see(Tkinter.END) 
		else:
			self.status_text.configure(text="File path invalid")
	 
	def export_to_jpg(self,file_path, name_only, new_directory):
		'''Opens file stream to PDF and streams each image found separted
		in the stream by the starmark and endmark
		Args:
		  file_path(str) = path to currrent PDF
		  name_only = 

		Returns:
		  none, saves to current directory
		'''
		pdf = file(file_path, "rb").read()
		startmark = "\xff\xd8"
		startfix = 0
		endmark = "\xff\xd9"
		endfix = 2
		i = 0
		njpg = 0
		while True:
			istream = pdf.find("stream", i)
			if istream < 0:
				break
			istart = pdf.find(startmark, istream, istream+20)
			if istart < 0:
				i = istream+20
				continue
			iend = pdf.find("endstream", istart)
			if iend < 0:
				raise Exception("Didn't find end of stream!")
			iend = pdf.find(endmark, iend-20)
			if iend < 0:
				raise Exception("Didn't find end of JPG!")
			istart += startfix
			iend += endfix
			jpg = pdf[istart:iend]
			jpgfile = file(new_directory+"%s_%d.jpg" % (name_only, njpg), "wb")
			jpgfile.write(jpg)
			jpgfile.close()
			njpg += 1
			i = iend
		return njpg


	def show_image(self, image_in):
		'''Displays image within GUI
		Args:
		  image_in(pil image) = image to display
		 
		Returns:
		  none, displays image
		'''
		pil_img = image_in.resize((100, 100), Image.ANTIALIAS) #The (250, 250) is (height, width)
		# convert to an image Tkinter can use
		tk_img = ImageTk.PhotoImage(pil_img)
		self.image_panel.configure(image=tk_img)
		self.image_panel.image = tk_img
		self.image_panel.pack(side = "bottom", fill = "both", expand = "yes")
		self.root.update_idletasks()


	def do_delete_PDF(self, file_path, name_only):
		'''Does the actual deleteion

			NOT IMPLEMENTED WITH LEARNING ALGORITHM YET

		Args:
		  image_in(pil image) = image to display
		 
		Returns:
		  none, displays image
		'''
		rsrcmgr = PDFResourceManager()
		retstr = StringIO()
		codec = 'utf-8'
		laparams = LAParams()
		device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
		fp = file(file_path, 'rb')
		interpreter = PDFPageInterpreter(rsrcmgr, device)
		password = ""
		maxpages = 0
		caching = True
		pagenos=set()

		for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
			interpreter.process_page(page)
			text = retstr.getvalue()
			print text[0]
		
		self.num_files_counter = self.num_files_counter+1
		text_string = 'current file (%s/%s): %s' %(self.num_files_counter,self.num_files,name_only)
		self.file_text.configure(text=text_string)
		input = PdfFileReader(file(file_path, "rb"))
		output_pdf = PdfFileWriter()
		deleted_pages = PdfFileWriter()
		self.output_field.insert(Tkinter.INSERT, '\n'+self.file_name)    
		pages_to_keep = []
		pages_to_delete = []
		content = ''
		for i in range(0,input.getNumPages()):
			if input.getPage(i).extractText():
				print input.getPage(i).extractText()
				pages_to_keep.append(str(i))
				output_pdf.addPage(input.getPage(i))
			else:
				pages_to_delete.append(str(i))
				deleted_pages.addPage(input.getPage(i))

		print pages_to_keep
		if len(pages_to_delete) > 0:
			log_display_format = "\n".join(pages_to_delete)
			self.output_field.insert(Tkinter.INSERT, '\nPAGES DELETED:\n')		
			self.output_field.insert(Tkinter.INSERT, '\n'+log_display_format)  
			outputfile = file(file_path+'_cleaned','wb')
			result.write(outputfile)
			outputfile.close()
			self.status_text.configure(text="Deleted pages")
		else:
			self.output_field.insert(Tkinter.INSERT, '\nNO PAGES DELETED\n')
			self.status_text.configure(text="No pages deleted")
		
root = Tkinter.Tk()
my_gui = Delete_PDF(root)
root.mainloop()