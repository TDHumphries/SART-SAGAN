
import argparse
from PIL import Image, ImageFilter
import random

import dataLoader

# def cleanTrainset(imgLoader, trainsetFolder = './trainset'):

# 	return imgLoader

def noisyTrainset(imgLoader, trainsetFolder = './trainset'):

	if imgLoader.loaderType != 'png':
		print('Error: imgTrainset.noisyTrainset() requires a PNG imgLoader.')
		print('       Please change the --dataType to png and provide a png format image dataset after --inputFolder')
		quit()

	outputPathList = []

	for imgFilePath in imgLoader.filePathList:
		image = Image.open(imgFilePath)
		image = image.filter(MyGaussianBlur(radius = 3))
		outputPath = trainsetFolder + '/noisy/' + imgFilePath.split('/')[-1].split('.')[0] + '_blur.png'
		image.save(outputPath)
		outputPathList.append(outputPath)

	blurImgLoader = dataLoader.PngLoader(pngFilePathList = outputPathList)
	return blurImgLoader


# def cleanEvaluationset(imgLoader, trainsetFolder = './trainset'):
# 	pass
# 	return imgLoader


# def noisyEvaluationset(imgLoader, trainsetFolder = './trainset'):
# 	pass
# 	return imgLoader


def createTrainsetPair(cleanImgLoader, noisyImgLoaderList):

	cleanImgFilePathList, noisyImgFilePathList, noisyImgLoaderType = [], [], None

	for noisyImgLoader in noisyImgLoaderList:
		if noisyImgLoader.loaderType != noisyImgLoaderType and noisyImgLoaderType != None:
			print('Error! In imgTrainset.createTrainsetPair()')
			print('       type of noisyImgLoader in noisyImgLoaderList not consistent.')
			print('       FATAL ERROR, FORCE EXIT.')
			exit()
		else:
			noisyImgLoaderType = noisyImgLoader.loaderType
		if len(cleanImgLoader) != len(noisyImgLoader):
			print('WARNING! In imgTrainset.createTrainsetPair()')
			print('         length of noisyImgLoader does not equal to length of cleanImgLoader')
			print('         len(noisyImgLoader) == %s, len(cleanImgLoader) == %s' % (str(len(noisyImgLoader)), str(len(cleanImgLoader))))
			if len(cleanImgLoader) > len(noisyImgLoader) and len(noisyImgLoader) > 0 and cleanImgLoader.filePathList[len(noisyImgLoader)-1].split('_')[0] == noisyImgLoader.filePathList[-1].split('_')[0]:
				print('         However, it seems that, noisyImgLoader is a subset of cleanImgLoader')
				print('         last file in noisyImgLoader is %s, we found %s in cleanImgLoader at same position.' % (str(noisyImgLoader.filePathList[-1]), str(cleanImgLoader.filePathList[len(noisyImgLoader)-1])))
				print('         a subset of cleanImgLoader will be used to utilize current noisyImgLoader')
				cleanImgFilePathList += cleanImgLoader.filePathList[:len(noisyImgLoader)]
				noisyImgFilePathList += noisyImgLoader.filePathList
			else:
				print('         We\'ll remove this noisyImgLoader from noisyImgLoaderList to avoid fatal error')
				noisyImgLoaderList.remove(noisyImgLoader)
		else:
			cleanImgFilePathList += cleanImgLoader.filePathList
			noisyImgFilePathList += noisyImgLoader.filePathList

	shuffleIndexList = list(range(len(cleanImgFilePathList)))
	random.shuffle(shuffleIndexList)
	cleanImgLoader = dataLoader.PngLoader() if cleanImgLoader.loaderType == 'png' else dataLoader.FltLoader()
	cleanImgLoader.filePathList = [ cleanImgFilePathList[shuffleIndex] for shuffleIndex in shuffleIndexList ]
	noisyImgLoader = dataLoader.PngLoader() if noisyImgLoaderList[0].loaderType == 'png' else dataLoader.FltLoader()
	noisyImgLoader.filePathList = [ noisyImgFilePathList[shuffleIndex] for shuffleIndex in shuffleIndexList ]

	return cleanImgLoader, noisyImgLoader




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


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--func', dest = 'func', type = str, default = 'default_func')
	parser.add_argument('--inFile', dest = 'inFile', type = str, default = 'default_in')
	parser.add_argument('--outFile', dest = 'outFile', type = str, default = 'default_out')
	parser.add_argument('--dataType', dest = 'dataType', type = str, default = 'png')
	parser.add_argument('--option', dest = 'option', type = str, default = 'default_option')
	args = parser.parse_args()

	print('args.inFile == %s' % str(args.inFile))

	if args.func == 'blur':

		image = Image.open(args.inFile)
		image = image.filter(MyGaussianBlur(radius = 3))
		image.save(args.outFile)
		print('Finish.')