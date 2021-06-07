
import argparse
import copy
# from tensorflow.python.client import device_lib

import dataLoader
import imgFormatConvert

def autoImgXCheck(imgX, info = ''):
	print('------------------------------------------------------')
	print('Check: debug.autoImgXCheck()  ' + info)
	print('       imgX.shape == %s' % str(imgX.shape))

	mat = imgX if len(imgX.shape) == 2 else imgFormatConvert.reshapeImgToMatrix(img = imgX) if len(imgX.shape) == 3 else imgFormatConvert.reshapeImgBatchToMatrix(imgBatch = imgX)
	flatMat = [ mat[i][j] for i in range(mat.shape[0]) for j in range(mat.shape[1]) ]

	print('       data type of pixel is: %s' % str(type(mat[0][0])))
	
	pixNum_minus = sum([ 1 if pixVal < 0 else 0 for pixVal in flatMat ])
	pixNum_0_0p2 = sum([ 1 if (pixVal > 0 and pixVal < 0.2) else 0 for pixVal in flatMat ])
	pixNum_0_1 = sum([ 1 if (pixVal > 0 and pixVal < 1) else 0 for pixVal in flatMat ])
	pixNum_2_255 = sum([ 1 if (pixVal > 2 and pixVal < 255) else 0 for pixVal in flatMat ])
	pixNum_255_ = sum([ 1 if pixVal > 255 else 0 for pixVal in flatMat ])
	print('       pixNum <0:0-0.2:0.2-1:2-255:>255 == %d:%d:%d:%d:%d' % (pixNum_minus, pixNum_0_0p2, pixNum_0_1 - pixNum_0_0p2, pixNum_2_255, pixNum_255_))
	if (pixNum_2_255 > len(flatMat) * 0.05): print('       255 level img')
	if (pixNum_0_1 - pixNum_0_0p2 > len(flatMat) * 0.05 and pixNum_2_255 <= len(flatMat) * 0.01): print('       float vlaue img')
	if (pixNum_0_0p2 == pixNum_0_1 and pixNum_2_255 == 0 and pixNum_255_ == 0): print('       black img')
	if (pixNum_255_ > len(flatMat) * 0.05): print('       img data overflow')
	if (pixNum_minus > 0): print('       minus value pixel exist')
	# print('------------------------------------------------------')

	return True if pixNum_255_ < 5 and pixNum_0_0p2 + pixNum_minus < len(flatMat) * 0.95 else False


def imgLoaderCheck(imgLoader, info = ''):

	print('------------------------------------------------------')
	print('Check: debug.dataLoaderCheck()  ' + info)
	if imgLoader is None:
		print('       imgLoader is None')
		return False
	print('       imgLoader list length == %d' % len(imgLoader.filePathList))
	if len(imgLoader) == 0:
		return False

	imgLoader = copy.deepcopy(imgLoader)
	img = imgLoader.loadImg()
	return autoImgXCheck(imgX = img)

def zzCopyImgLoader(imgLoader = None):

	newImgLoader = dataLoader.PngLoader(pngFilePathList = imgLoader.filePathList) if imgLoader.loaderType == 'png' else dataLoader.FltLoader(fltFilePathList = imgLoader.filePathList) if imgLoader.loaderType == 'flt' else None

	return newImgLoader

def quickCheck(imgX):
	img = imgX if len(imgX.shape) == 2 else imgFormatConvert.reshapeImgToMatrix(img = imgX) if len(imgX.shape) == 3 else imgFormatConvert.reshapeImgBatchToMatrix(imgBatch = imgX)

	y0, y1, y2, y3 = int(img.shape[0]*1/6), int(img.shape[0]/4), int(img.shape[0]/2), int(img.shape[0]*3/4)
	x0, x1, x2, x3 = int(img.shape[1]*1/6), int(img.shape[1]/4), int(img.shape[1]/2), int(img.shape[1]*3/4)
	pixelValueSampleList = [img[y0][x0], img[y0][x1], img[y0][x2], img[y0][x3], 
							img[y1][x0], img[y1][x1], img[y1][x2], img[y1][x3], 
							img[y2][x0], img[y2][x1], img[y2][x2], img[y2][x3], 
							img[y3][x0], img[y3][x1], img[y3][x2], img[y3][x3]]
	whetherDataOverflow = sum([ 1 if pixVal > 256 else 0 for pixVal in pixelValueSampleList ]) > 1
	whetherBlackImage = sum([ 1 if pixVal > 0.01 else 0 for pixVal in pixelValueSampleList ]) < 1
	whether255 = True in [ pixVal > 5 for pixVal in pixelValueSampleList ]

	return 'data_overflow' if whetherDataOverflow else 'black' if whetherBlackImage else '255' if whether255 else 'float'

def reportError(description, position = 'Unknow position', fatal = False):
	print('Error! In %s' % str(position))
	if type(description) is str:
		print('       %s' % description)
	elif type(description) is list:
		for line in description:
			print('       %s' % line)
	else:
		print('       %s' % str(description))
	if fatal is True:
		print('       FATAL ERROR, FORCE EXIT')
		exit()
	return None

def pairLoaderCheck(imgLoader_1, imgLoader_2):
	imgLoader_1, imgLoader_2 = copy.deepcopy(imgLoader_1), copy.deepcopy(imgLoader_2)

	if len(imgLoader_1) != len(imgLoader_2) or len(imgLoader_1) == 0 or len(imgLoader_2) == 0:
		print('Error! In debug.pairLoaderCheck()')
		print('       len(imgLoader_1) == %s; len(imgLoader_2) == %s' % (str(len(imgLoader_1)), str(len(imgLoader_2))))
		exit()

	i = 0
	for (_, pureName_1), (_, pureName_2) in zip(imgLoader_1, imgLoader_2):
		i += 1
		if pureName_1.split('_')[0] != pureName_2.split('_')[0]:
			print('Error! In debug.pairLoaderCheck()')
			print('       pureName_1 == %s, pureName_2 == %s' % (str(pureName_1), str(pureName_2)))
			exit()
		if i % 1000 == 0:
			print('Info: debug.pairLoaderCheck(): %d img checked' % i)

	print('Info: debug.pairLoaderCheck(): Success!')


# def patch_GPUdistribution_for_NNstructure_discriminatorNN_simpleGAN():
# 	local_device_protos = device_lib.list_local_devices()
# 	avaliableGPUlist = [ x.name.replace('device:', '').lower() for x in local_device_protos if 'GPU' in x.name ]
# 	print('Info: In debug.patch_GPUdistribution_for_NNstructure_discriminatorNN_simpleGAN()')
# 	print('      avaliable GPU: %s' % str(avaliableGPUlist))
# 	useTrainingDistribution = True if len(avaliableGPUlist) >= 4 else False
# 	return avaliableGPUlist[1] if useTrainingDistribution else avaliableGPUlist[0]



def detailedImgXCheck(imgX, info = ''):
	print('------------------------------------------------------')
	print('Check: debug.detailedImgXCheck()  ' + info)
	print('       imgX.shape == %s' % str(imgX.shape))

	mat = imgX if len(imgX.shape) == 2 else imgFormatConvert.reshapeImgToMatrix(img = imgX) if len(imgX.shape) == 3 else imgFormatConvert.reshapeImgBatchToMatrix(imgBatch = imgX)
	flatMat = [ mat[i][j] for i in range(mat.shape[0]) for j in range(mat.shape[1]) ]

	print('       data type of pixel is: %s' % str(type(mat[0][0])))

	startList = [0, 1] + list(range(255))[::10][1:]
	endList = [1] + list(range(255))[::10][1:] + [255]
	zzList = []

	for st, ed in zip(startList, endList):
		zzList.append(('pixVal in '+str(st)+'-'+str(ed), sum([ 1 if (pixVal >= st and pixVal < ed) else 0 for pixVal in flatMat ])))

	print('       pixVal < 0:', sum([ 1 if pixVal < 0 else 0 for pixVal in flatMat ]))
	for zz in zzList:
		print('       ' + zz[0] + ':', zz[1])
	print('       pixVal = 255:', sum([ 1 if pixVal == 255 else 0 for pixVal in flatMat ]))
	print('       pixVal > 255:', sum([ 1 if pixVal > 255 else 0 for pixVal in flatMat ]))

	
	pixNum_minus = sum([ 1 if pixVal < 0 else 0 for pixVal in flatMat ])
	pixNum_0_0p2 = sum([ 1 if (pixVal >= 0 and pixVal < 0.2) else 0 for pixVal in flatMat ])
	pixNum_0_1 = sum([ 1 if (pixVal > 0 and pixVal <= 1) else 0 for pixVal in flatMat ])
	pixNum_1_2 = sum([ 1 if (pixVal > 1 and pixVal < 2) else 0 for pixVal in flatMat ])
	pixNum_2_255 = sum([ 1 if (pixVal > 2 and pixVal < 255) else 0 for pixVal in flatMat ])
	pixNum_255_ = sum([ 1 if pixVal > 255 else 0 for pixVal in flatMat ])
	# print('       pixNum <0:0-0.2:0.2-1:2-255:>255 == %d:%d:%d:%d:%d' % (pixNum_minus, pixNum_0_0p2, pixNum_0_1 - pixNum_0_0p2, pixNum_2_255, pixNum_255_))
	if (pixNum_2_255 > len(flatMat) * 0.05): print('       255 level img')
	if (pixNum_0_1 - pixNum_0_0p2 > len(flatMat) * 0.05 and pixNum_2_255 <= len(flatMat) * 0.01): print('       float vlaue img')
	if (pixNum_0_0p2 == pixNum_0_1 and pixNum_2_255 == 0 and pixNum_255_ == 0): print('       black img')
	if (pixNum_255_ > len(flatMat) * 0.05): print('       img data overflow')
	if (pixNum_minus > 0): print('       minus value pixel exist')
	# print('------------------------------------------------------')

	return True if pixNum_255_ < 5 and pixNum_0_0p2 + pixNum_minus < len(flatMat) * 0.95 else False








if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--func', dest = 'func', type = str, default = 'default_func')
	parser.add_argument('--inFile', dest = 'inFile', type = str, default = 'default_in')
	parser.add_argument('--outFile', dest = 'outFile', type = str, default = 'default_out')
	parser.add_argument('--dataType', dest = 'dataType', type = str, default = 'png')
	args = parser.parse_args()

	print('args.inFile == %s' % str(args.inFile))

	if args.func == 'test':
		imgLoader = dataLoader.PngLoader(pngFilePathList = [args.inFile, args.inFile, args.inFile])
		img = imgLoader.loadImg()
		a = autoImgXCheck(imgX = img)
		b = autoImgXCheck(imgX = img * 255)
		c = autoImgXCheck(imgX = img / 255)
		d = autoImgXCheck(imgX = img * 255 * 255)
		e = autoImgXCheck(imgX = -img)
		f = imgLoaderCheck(imgLoader = imgLoader)
		g = quickCheck(imgX = img)
		h = quickCheck(imgX = img * 255)
		i = quickCheck(imgX = img / 255)
		j = quickCheck(imgX = img * 255 * 255)
		print(a, b, c, d, e, f, g, h, i, j)

	elif args.func == 'checkImg':
		imgLoader = dataLoader.PngLoader(pngFilePathList = [args.inFile])
		imgLoaderCheck(imgLoader = imgLoader)



