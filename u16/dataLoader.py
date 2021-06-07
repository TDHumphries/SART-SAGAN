
import argparse
import glob
import numpy as np
from PIL import Image

import imgPainter
import debug
import parameters


class FltLoader(object):

	def __init__(self, fltFilePathList = None, sort = True):
		self.loaderType = 'flt'
		self.filePathList = fltFilePathList if fltFilePathList is not None else []
		if sort is True: self.filePathList.sort()

	def __len__(self):
		return len(self.filePathList)

	def __iter__(self):
		return self

	def __next__(self):
		img, pureName = self.loadImg_withName()
		if img is not None:
			return img, pureName
		else:
			raise StopIteration
		
	def next(self):
		return self.__next__()

	def __getitem__(self, index):
		if type(index) is int:
			return self.loadImg_specific(imgFilePath = self.filePathList[index]), self.filePathList[index].split('/')[-1].split('.')[0]
		elif type(index) is slice:
			return FltLoader(fltFilePathList = self.filePathList[index], sort = False)
		else:
			print('WARNING! In dataLoader.FltLoader.__getitem__()')
			print('         unrecongnize index type: %s' % str(type(index)))
			print('         __getitem__() will return None')
			return None

	def __reshapeToImg(self, data):
		if len(data.shape) == 1 and data.shape[0] == 512 * 512:
			return np.reshape(a = data, newshape = (512, 512, 1))
		elif len(data.shape) == 1 and data.shape[0] == 900 * 729:
			return np.reshape(a = data, newshape = (900, 729, 1))
		elif len(data.shape) == 1 and data.shape[0] % 729 == 0:
			print('WARNING! in dataLoader.FltLoader.__reshapeToImg()')
			print('         Unknow flt image size; data.shape == %s' % str(data.shape))
			print('         We\'ll try to reshape it as %s * 729 image' % str(data.shape[0] / 729))
			return np.reshape(a = data, newshape = (int(data.shape[0] / 729), 729, 1))
		else:
			print('ERROR: in dataLoader.FltLoader.__reshapeToImg()')
			print('       cannot reshape data to img')
			print('       data.shape == %s' % str(data.shape))
		pass

	def loadImg(self):
		if len(self.filePathList) == 0:
			return None
		filePath = self.filePathList[0]
		self.filePathList = self.filePathList[1: ]
		return self.__reshapeToImg(data = np.fromfile(file = filePath, dtype = 'float32', count = -1))

	def loadImg_withName(self):
		if len(self.filePathList) == 0:
			return None, None
		filePureName = self.filePathList[0].split('/')[-1].split('.')[0]
		return self.loadImg(), filePureName


	def loadImgBatch(self, batchSize):
		if batchSize > len(self.filePathList):
			return None
		tempFilePathList = self.filePathList[ :batchSize]
		self.filePathList = self.filePathList[batchSize: ]
		imgList = [ self.__reshapeToImg(data = np.fromfile(file = filePath, dtype = 'float32', count = -1)) for filePath in tempFilePathList ]
		return np.array(imgList)

	def loadImg_specific(self, imgFilePath):
		return self.__reshapeToImg(data = np.fromfile(file = imgFilePath, dtype = 'float32', count = -1))

	def popBatchAsImgLoader(self, batchSize):
		batchSize = min(int(batchSize), len(self.filePathList))
		if batchSize == 0:
			return None
		newImgLoader = FltLoader(fltFilePathList = self.filePathList[:batchSize], sort = False)
		self.filePathList = self.filePathList[batchSize:]
		return newImgLoader

	def loadImgPatch(self, sidelength_X = 128, sidelength_Y = 128, batchSize = 1):
		imgList = []
		for imgNum in range(batchSize):
			img = self.loadImg()
			for i in range(int(img.shape[0]/sidelength_X)):
				for j in range(int(img.shape[1]/sidelength_Y)):
					imgList.append(img[sidelength_X*i:sidelength_X*(i+1),sidelength_Y*j:sidelength_Y*(j+1)])
		return np.array(imgList)

	def loadImgPatch_withName(self, sidelength_X = 128, sidelength_Y = 128):
		imgList = []
		img, imgPureName = self.loadImg_withName()
		for i in range(int(img.shape[0]/sidelength_X)):
			for j in range(int(img.shape[1]/sidelength_Y)):
				imgList.append(img[sidelength_X*i:sidelength_X*(i+1),sidelength_Y*j:sidelength_Y*(j+1)])
		return np.array(imgList), imgPureName
		


class PngLoader(object):

	def __init__(self, pngFilePathList = None, sort = True):
		self.loaderType = 'png'
		self.filePathList = pngFilePathList if pngFilePathList is not None else []
		if sort is True: self.filePathList.sort()

	def __len__(self):
		return len(self.filePathList)

	def __iter__(self):
		return self

	def __next__(self):
		img, pureName = self.loadImg_withName()
		if img is not None:
			return img, pureName
		else:
			raise StopIteration

	def next(self):
		return self.__next__()

	def __getitem__(self, index):
		if type(index) is int:
			return self.loadImg_specific(imgFilePath = self.filePathList[index]), self.filePathList[index].split('/')[-1].split('.')[0]
		elif type(index) is slice:
			return PngLoader(pngFilePathList = self.filePathList[index], sort = False)
		else:
			print('WARNING! In dataLoader.PngLoader.__getitem__()')
			print('         unrecongnize index type: %s' % str(type(index)))
			print('         __getitem__() will return None')
			return None

	def loadImg(self):
		if len(self.filePathList) == 0:
			return None
		filePath = self.filePathList[0]
		self.filePathList = self.filePathList[1: ]
		im = Image.open(filePath).convert('L')
		img = np.array(im).reshape(im.size[1], im.size[0], 1)
		return img.astype(float) / 255

	def loadImg_withName(self):
		if len(self.filePathList) == 0:
			return None, None
		filePureName = self.filePathList[0].split('/')[-1].split('.')[0]
		return self.loadImg(), filePureName

	def loadImgBatch(self, batchSize):
		if batchSize > len(self.filePathList):
			return None
		tempFilePathList = self.filePathList[ :batchSize]
		self.filePathList = self.filePathList[batchSize: ]
		batch = []
		for filePath in tempFilePathList:
			im = Image.open(filePath).convert('L')
			batch.append(np.array(im).reshape(im.size[1], im.size[0], 1))
		imgBatch = np.array(batch)
		return imgBatch.astype(float) / 255

	def loadImg_specific(self, imgFilePath):
		im = Image.open(imgFilePath).convert('L')
		return np.array(im).reshape(im.size[1], im.size[0], 1)

	def popBatchAsImgLoader(self, batchSize):
		batchSize = min(int(batchSize), len(self.filePathList))
		if batchSize == 0:
			return None
		newImgLoader = PngLoader(pngFilePathList = self.filePathList[:batchSize], sort = False)
		self.filePathList = self.filePathList[batchSize:]
		return newImgLoader

	def loadImgPatch(self, sidelength_X = 128, sidelength_Y = 128, batchSize = 1):
		imgList = []
		for imgNum in range(batchSize):
			img = self.loadImg()
			for i in range(int(img.shape[0]/sidelength_X)):
				for j in range(int(img.shape[1]/sidelength_Y)):
					imgList.append(img[sidelength_X*i:sidelength_X*(i+1),sidelength_Y*j:sidelength_Y*(j+1)])
		return np.array(imgList)
		
	def loadImgPatch_withName(self, sidelength_X = 128, sidelength_Y = 128):
		imgList = []
		img, imgPureName = self.loadImg_withName()
		for i in range(int(img.shape[0]/sidelength_X)):
			for j in range(int(img.shape[1]/sidelength_Y)):
				imgList.append(img[sidelength_X*i:sidelength_X*(i+1),sidelength_Y*j:sidelength_Y*(j+1)])
		return np.array(imgList), imgPureName


def folderPathToImgLoader(floderPath, dataType):
	filePathList = glob.glob(floderPath + '/*.' + dataType) if floderPath[-1] != '/' else glob.glob(floderPath + '*.' + dataType)
	return PngLoader(pngFilePathList = filePathList) if dataType == 'png' else FltLoader(fltFilePathList = filePathList) if dataType == 'flt' else debug.reportError(description = 'unknow dataType %s' % str(dataType), position = 'dataLoader.folderPathToImgLoader()', fatal = True)

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--func', dest = 'func', type = str, default = 'default_func')
	parser.add_argument('--inFile', dest = 'inFile', type = str, default = 'default_in')
	parser.add_argument('--outFile', dest = 'outFile', type = str, default = 'default_out')
	args = parser.parse_args()

	print('args.inFile == %s' % str(args.inFile))

	if args.func == 'testFlt':
		imgLoader = FltLoader(fltFilePathList = [args.inFile, args.inFile, args.inFile, args.inFile, args.inFile])
	elif args.func == 'testPng':
		imgLoader = PngLoader(pngFilePathList = [args.inFile, args.inFile, args.inFile, args.inFile, args.inFile])

	print('imgLoader.filePathList == %s' % str(imgLoader.filePathList))
	print('lenth of imgLoader is: %s' % str(len(imgLoader)))
	img = imgLoader.loadImg()
	# print(img)
	imgPainter.autoPaint(imgX = img, path = args.outFile)

	print('lenth of imgLoader is: %s' % str(len(imgLoader)))
	img, imgPureName = imgLoader.loadImg_withName()
	print('imgPureName == %s' % str(imgPureName))

	print('lenth of imgLoader is: %s' % str(len(imgLoader)))
	imgBatch = imgLoader.loadImgBatch(batchSize = 2)
	print('shape of imgBatch is: %s' % str(imgBatch.shape))

	img = imgLoader.loadImg_specific(imgFilePath = args.inFile)
	print('shape of img from loadImg_specific() is: %s' % str(img.shape))
