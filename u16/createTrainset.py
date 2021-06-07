

import argparse
import astra
import glob
import numpy as np
import os
from PIL import Image, ImageFilter

import dataLoader
import debug
import imgFormatConvert
import imgPainter
import parameters
import reconstruction
import sinogram



class MyGaussianBlur(ImageFilter.Filter):
	name = "GaussianBlur"

	def __init__(self, radius = 2, bounds = None):
		self.radius = radius
		self.bounds = bounds

	def filter(self, image):
		if self.bounds:
			clips = image.crop(self.bounds).gaussian_blur(self.radius)
			image.paste(clips, self.bounds)
			return image
		else:
			return image.gaussian_blur(self.radius)


def createGuassianBlur(imgLoader, outputFolderPath, radius):

	inputFilePathList = imgLoader.filePathList
	outputPathList = []
	
	for imgFilePath in inputFilePathList:
		image = Image.open(imgFilePath)
		image = image.filter(MyGaussianBlur(radius = radius))
		outputPath = outputFolderPath + '/' + imgFilePath.split('/')[-1].split('.')[0] + '_blur_r%d.png' % radius
		image.save(outputPath)
		outputPathList.append(outputPath)

	blurImgLoader = dataLoader.PngLoader(pngFilePathList = outputPathList)

	return blurImgLoader


def createReconBlur(imgLoader, outputFolderPath, option, iterNum, ns, tempFolderPath = '../temp'):

	print('Debug: In createTrainset.createReconBlur() option == %s' % option)

	imgLoaderInput = imgLoader
	numEachLoad = 10
	while len(imgLoaderInput.filePathList) > 0:
		(newPathList, imgLoaderInput.filePathList) = (imgLoaderInput.filePathList[:numEachLoad], imgLoaderInput.filePathList[numEachLoad:]) if len(imgLoaderInput.filePathList) > numEachLoad else (imgLoaderInput.filePathList, [])
		imgLoader = dataLoader.PngLoader(pngFilePathList = newPathList) if imgLoaderInput.loaderType == 'png' else dataLoader.FltLoader(fltFilePathList = newPathList)

		debug.imgLoaderCheck(imgLoader = imgLoader)

		_, sinogramImgLoader = sinogram.createSinogram_batch(imgLoader = imgLoader, sinogramFolderPath = tempFolderPath, option = option)

		_, reconImgLoader = reconstruction.sart_batch(sinogramImgLoader = sinogramImgLoader, outputFolderPath = outputFolderPath, NNfunction = None, option = option, iterNum = iterNum, paintSteps = False, ns = ns)
		# _, reconImgLoader = reconstruction.sart_batch(sinogramImgLoader = sinogramImgLoader, outputFolderPath = args.outFile, NNfunction = None, option = args.option, iterNum = 200)

	return reconImgLoader


def trainsetAugmentation(oriImgFolderPath, outputFolderPath, dataType = 'png'):
	print('Info: createTrainset.trainsetAugmentation() start.')
	
	oriImgFilePathList = glob.glob(oriImgFolderPath + '/*.' + dataType)
	oriImgLoader = dataLoader.PngLoader(pngFilePathList = oriImgFilePathList) if dataType == 'png' else dataLoader.FltLoader(fltFilePathList = oriImgFilePathList) if dataType == 'flt' else None
	i = 0
	for oriImg, oriImgPureName in oriImgLoader:
		i += 1
		print('    image %d' % i)
		mat = imgFormatConvert.reshapeImgToMatrix(oriImg)
		mat_rot90 = np.rot90(m = mat, k = 1)
		mat_rot180 = np.rot90(m = mat, k = 2)
		mat_rot270 = np.rot90(m = mat, k = 3)

		imgPainter.autoPaint(imgX = mat_rot90, path = outputFolderPath + '/augR90_' + oriImgPureName + '.png', reportImageInfo = False)
		imgPainter.autoPaint(imgX = mat_rot180, path = outputFolderPath + '/augR180_' + oriImgPureName + '.png', reportImageInfo = False)
		imgPainter.autoPaint(imgX = mat_rot270, path = outputFolderPath + '/augR270_' + oriImgPureName + '.png', reportImageInfo = False)
		imgPainter.autoPaint(imgX = mat.T, path = outputFolderPath + '/augT_' + oriImgPureName + '.png', reportImageInfo = False)
	print('Image Rotate Finish.')


def dataAugmentation(inputImgLoader, outputFolderPath = '../temp', outputDataType = 'png'):

	imgList = []

	for oriImg, oriImgPureName in inputImgLoader:
		print('Info: Data Augmentation - %s' % str(oriImgPureName))
		mat = imgFormatConvert.reshapeImgToMatrix(oriImg)
		mat_rot90 = np.rot90(m = mat, k = 1)
		mat_rot180 = np.rot90(m = mat, k = 2)
		mat_rot270 = np.rot90(m = mat, k = 3)

		if outputDataType == 'png':
			imgPainter.autoPaint(imgX = mat_rot90, path = outputFolderPath + '/' + oriImgPureName + '-R90.png', reportImageInfo = False)
			imgPainter.autoPaint(imgX = mat_rot180, path = outputFolderPath + '/' + oriImgPureName + '-R180.png', reportImageInfo = False)
			imgPainter.autoPaint(imgX = mat_rot270, path = outputFolderPath + '/' + oriImgPureName + '-R270.png', reportImageInfo = False)
			imgPainter.autoPaint(imgX = mat.T, path = outputFolderPath + '/' + oriImgPureName + '-RT.png', reportImageInfo = False)
			imgPainter.autoPaint(imgX = mat, path = outputFolderPath + '/' + oriImgPureName + '-Rori.png', reportImageInfo = False)

			imgList += [outputFolderPath + '/' + oriImgPureName + '-R90.png', \
						outputFolderPath + '/' + oriImgPureName + '-R180.png', \
						outputFolderPath + '/' + oriImgPureName + '-R270.png', \
						outputFolderPath + '/' + oriImgPureName + '-RT.png', \
						outputFolderPath + '/' + oriImgPureName + '-Rori.png']
		else:
			print('ERROR! In createTrainset.dataAugmentation()')
			print('       Unsupported output data type:', outputDataType)
			print('       FORCE EXIT')
			exit()

	return dataLoader.PngLoader(pngFilePathList = imgList) if outputDataType == 'png' else dataLoader.FltLoader(fltFilePathList = imgList) if outputDataType == 'flt' else None

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--func', dest = 'func', type = str, default = 'full')
	parser.add_argument('--inFile', dest = 'inFile', type = str, default = '../NDCT')
	parser.add_argument('--outFile', dest = 'outFile', type = str, default = '../uiharuDataset/dataset719_la140ns50it20')
	parser.add_argument('--dataType', dest = 'dataType', type = str, default = 'png')
	parser.add_argument('--option', dest = 'option', type = str, default = 'limited_angle_140')
	parser.add_argument('--radius', dest = 'radius', type = int, default = 2)
	parser.add_argument('--iterNum', dest = 'iterNum', type = int, default = 20)
	parser.add_argument('--maxImg', dest = 'maxImg', type = int, default = None)
	parser.add_argument('--ns', dest = 'ns', type = int, default = 50)
	parser.add_argument('--handleNum', dest = 'handleNum', type = int, default = 10)
	parser.add_argument('--startFrom', dest = 'startFrom', type = int, default = 0)
	args = parser.parse_args()

	print('Info: args.inFile == %s' % str(args.inFile))

	if not os.path.exists(args.inFile):
		print('ERROR: args.inFile not exist')
		print('       Force exit.')
		exit()
	if not os.path.exists(args.outFile):
		os.mkdir(args.outFile)
		os.mkdir(args.outFile+'/steps')
		os.mkdir(args.outFile+'/sino')
		os.mkdir(args.outFile+'/augCleanCT')
		


	# parameters.globalParametersInitialization(option = args.option)

	if args.func == 'full':
		imgLoader = dataLoader.PngLoader(pngFilePathList = glob.glob(args.inFile + '/*.png')) if args.dataType == 'png' else dataLoader.FltLoader(fltFilePathList = glob.glob(args.inFile + '/*.flt')) if args.dataType == 'flt' else None
		imgLoader = imgLoader[args.startFrom:]
		print('Start generate images from: %s' % str(imgLoader.filePathList[0]))
		while len(imgLoader) > 0:
			batchImgLoader = imgLoader.popBatchAsImgLoader(batchSize = args.handleNum)
			augmentedBatchImgLoader = dataAugmentation(inputImgLoader = batchImgLoader, outputFolderPath = args.outFile+'/augCleanCT', outputDataType = args.dataType)

			optionList = args.option.split(';')
			for tempOption in optionList:
				if not os.path.exists(args.outFile+'/augNoisyCT_'+str(tempOption)):
					os.mkdir(args.outFile+'/augNoisyCT_'+str(tempOption))

				_, sinogramImgLoader = sinogram.createSinogram_batch(imgLoader = augmentedBatchImgLoader[:], sinogramFolderPath = args.outFile+'/sino', option = tempOption)
				debug.imgLoaderCheck(imgLoader = sinogramImgLoader)

				reconImgLoader = reconstruction.pureSART_batch(sinogramImgLoader = sinogramImgLoader, outputFolderPath = args.outFile+'/augNoisyCT_'+str(tempOption), noiseOption = tempOption, iterNum = args.iterNum, ns = args.ns)
				debug.imgLoaderCheck(imgLoader = reconImgLoader)
				# _, reconImgLoader_SART = reconstruction.MNNsart_batch(sinogramImgLoader = sinogramImgLoader, outputFolderPath = args.outFile+'/augNoisyCT_'+str(tempOption), option = tempOption, mnnOrder = 'sart199|return', paintSteps = False, imgMiddleName = '_SART', ns = 10)
	

	elif args.func == 'normal':
		imgLoader = dataLoader.PngLoader(pngFilePathList = glob.glob(args.inFile + '/*.png')) if args.dataType == 'png' else dataLoader.FltLoader(fltFilePathList = glob.glob(args.inFile + '/*.flt')) if args.dataType == 'flt' else None
		imgLoader = imgLoader[args.startFrom:]
		print('Start generate images from: %s' % str(imgLoader.filePathList[0]))
		while len(imgLoader) > 0:
			batchImgLoader = imgLoader.popBatchAsImgLoader(batchSize = args.handleNum)
			augmentedBatchImgLoader = batchImgLoader #dataAugmentation(inputImgLoader = batchImgLoader, outputFolderPath = args.outFile+'/augCleanCT', outputDataType = args.dataType)

			optionList = args.option.split(';')
			for tempOption in optionList:
				if not os.path.exists(args.outFile+'/'+str(tempOption)):
					os.mkdir(args.outFile+'/'+str(tempOption))
				
				_, sinogramImgLoader = sinogram.createSinogram_batch(imgLoader = augmentedBatchImgLoader[:], sinogramFolderPath = args.outFile+'/sino', option = tempOption)
				debug.imgLoaderCheck(imgLoader = sinogramImgLoader)

				reconImgLoader = reconstruction.pureSART_batch(sinogramImgLoader = sinogramImgLoader, outputFolderPath = args.outFile+'/'+str(tempOption), noiseOption = tempOption, iterNum = args.iterNum, ns = args.ns)
				debug.imgLoaderCheck(imgLoader = reconImgLoader)
				# _, reconImgLoader_SART = reconstruction.MNNsart_batch(sinogramImgLoader = sinogramImgLoader, outputFolderPath = args.outFile+'/augNoisyCT_'+str(tempOption), option = tempOption, mnnOrder = 'sart199|return', paintSteps = False, imgMiddleName = '_SART', ns = 10)
		





	# elif args.func == 'gaussian_blur':

	# 	imgLoader = dataLoader.PngLoader(pngFilePathList = glob.glob(args.inFile + '/*.png'))
	# 	blurImgLoader = createGuassianBlur(imgLoader = imgLoader, outputFolderPath = args.outFile, radius = args.radius)
	# 	debug.imgLoaderCheck(imgLoader = blurImgLoader)

	# elif args.func == 'sparse_view_180_blur':

	# 	pngFilePathList = glob.glob(args.inFile + '/*.png')
	# 	pngFilePathList.sort()
	# 	imgLoader = dataLoader.PngLoader(pngFilePathList = pngFilePathList[:1000])
	# 	reconImgLoader = createReconBlur(imgLoader = imgLoader, outputFolderPath = args.outFile, option = 'sparse_view_180', iterNum = args.iterNum)
	# 	debug.imgLoaderCheck(imgLoader = reconImgLoader)

	# elif args.func == 'recon_blur':
	# 	pngFilePathList = glob.glob(args.inFile + '/*.png')
	# 	pngFilePathList.sort()
	# 	imgLoader = dataLoader.PngLoader(pngFilePathList = pngFilePathList[:min(args.maxImg, len(pngFilePathList))]) if args.maxImg is not None else dataLoader.PngLoader(pngFilePathList = pngFilePathList)
	# 	reconImgLoader = createReconBlur(imgLoader = imgLoader, outputFolderPath = args.outFile, option = args.option, iterNum = args.iterNum, tempFolderPath = args.outFile + '/sino', ns = args.ns)
	# 	debug.imgLoaderCheck(imgLoader = reconImgLoader)

	# elif args.func == 'aug_data':
	# 	trainsetAugmentation(oriImgFolderPath = args.inFile, outputFolderPath = args.outFile, dataType = 'png')

	# elif args.func.find('recon_blur-') == 0:
	# 	a, b = int(args.func[len('recon_blur-'):].split(',')[0]), int(args.func[len('recon_blur-'):].split(',')[1])
	# 	pngFilePathList = glob.glob(args.inFile + '/*.png')
	# 	pngFilePathList.sort()
	# 	if b == -1:
	# 		b = len(pngFilePathList)
	# 	print('a, b == %d, %d' % (a, b))
	# 	imgLoader = dataLoader.PngLoader(pngFilePathList = pngFilePathList[a:b])
	# 	reconImgLoader = createReconBlur(imgLoader = imgLoader, outputFolderPath = args.outFile, option = args.option, iterNum = args.iterNum, tempFolderPath = args.outFile + '/sino', ns = args.ns)
	# 	debug.imgLoaderCheck(imgLoader = reconImgLoader)

