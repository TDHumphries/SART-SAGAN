
import argparse
import numpy as np
import scipy as sp
import scipy.misc

import imgFormatConvert

def paintMatrix(matrix, path = './temp/defaultPaintPath.jpg'):
	sp.misc.imsave(name = path, arr = matrix.astype('uint8'))


def fixMatrixDataOverflow(matrix = None):
	
	for i in range(len(matrix)):
		for j in range(len(matrix[i])):
			matrix[i][j] = matrix[i][j] if matrix[i][j] >= 0 else 0
	return matrix


def autoPaint(imgX, path = './temp/defaultPaintPath.jpg', reportImageInfo = True, fixDataOverflow = False):

	img = imgX if len(imgX.shape) == 2 else imgFormatConvert.reshapeImgToMatrix(img = imgX) if len(imgX.shape) == 3 else imgFormatConvert.reshapeImgBatchToMatrix(imgBatch = imgX)

	y0, y1, y2, y3 = int(img.shape[0]*1/6), int(img.shape[0]/4), int(img.shape[0]/2), int(img.shape[0]*3/4)
	x0, x1, x2, x3 = int(img.shape[1]*1/6), int(img.shape[1]/4), int(img.shape[1]/2), int(img.shape[1]*3/4)
	pixelValueSampleList = [img[y0][x0], img[y0][x1], img[y0][x2], img[y0][x3], 
							img[y1][x0], img[y1][x1], img[y1][x2], img[y1][x3], 
							img[y2][x0], img[y2][x1], img[y2][x2], img[y2][x3], 
							img[y3][x0], img[y3][x1], img[y3][x2], img[y3][x3]]
	sample255TruthList = [ pix > 5 for pix in pixelValueSampleList]
	matrix = img if True in sample255TruthList else img * 255

	if True in sample255TruthList and reportImageInfo is True:
		print('Info: imgPainter.autoPaint() is painting %s' % path)
		print('      255 level image, size: %s' % str(imgX.shape))
	elif reportImageInfo is True:
		print('Info: imgPainter.autoPaint() is painting %s' % path)
		print('      float value image, size: %s' % str(imgX.shape))

	matrix = fixMatrixDataOverflow(matrix = matrix) if fixDataOverflow is True else matrix
	paintMatrix(matrix = matrix, path = path)


def concatImg(imgXList):
	return np.concatenate([ np.squeeze(np.clip(255*imgX, 0, 255).astype('uint8')) for imgX in imgXList ], axis = 1)


def paintFltFileToPng(fltFilePath, outputFilePath):
	
	data = np.fromfile(file = fltFilePath, dtype = 'float32', count = -1)
	if data.shape[0] == 900 * 729:
		reshapeToImg = lambda data: np.reshape(a = data, newshape = (900, 729, 1))
		print('Info: imgPainter.paintFltFileToPng(): reshape data as sinogram image.')
	elif data.shape[0] == 512 * 512:
		reshapeToImg = lambda data: np.reshape(a = data, newshape = (512, 512, 1))
		print('Info: imgPainter.paintFltFileToPng(): reshape data as 512 * 512 image.')
	else:
		print('Error: imgPainter.paintFltFileToPng(): unknow image shape, data.shape == %s' % str(data.shape))
		print('       Forced EXIT.')
		exit()

	img = reshapeToImg(data = np.fromfile(file = fltFilePath, dtype = 'float32', count = -1))
	autoPaint(imgX = img, path = outputFilePath, reportImageInfo = True)


'''=============================================================================================================='''
'''=============================================================================================================='''
'''=============================================================================================================='''
'''=============================================================================================================='''

def pickChannelX(img, channelX = 0):
	imgNew = []
	for i in range(len(img)):
		imgNew.append([])
		for j in range(len(img[i])):
			imgNew[i].append(img[i][j][channelX])
	return np.array(imgNew)

def autoPaintPlus(imgX, path = './temp/defaultPaintPath.jpg', channel = 0, reportImageInfo = True, fixDataOverflow = False):

	if len(imgX.shape) == 4:
		img3 = imgX[0]
	elif len(imgX.shape) == 3:
		img3 = imgX
	
	if len(imgX.shape) != 2:
		imgX = pickChannelX(img = img3, channelX = channel)
		# print('hhhhh:', imgX.shape)

	img = imgX if len(imgX.shape) == 2 else imgFormatConvert.reshapeImgToMatrix(img = imgX) if len(imgX.shape) == 3 else imgFormatConvert.reshapeImgBatchToMatrix(imgBatch = imgX)

	# y0, y1, y2, y3 = int(img.shape[0]*1/6), int(img.shape[0]/4), int(img.shape[0]/2), int(img.shape[0]*3/4)
	# x0, x1, x2, x3 = int(img.shape[1]*1/6), int(img.shape[1]/4), int(img.shape[1]/2), int(img.shape[1]*3/4)
	# pixelValueSampleList = [img[y0][x0], img[y0][x1], img[y0][x2], img[y0][x3], 
	# 						img[y1][x0], img[y1][x1], img[y1][x2], img[y1][x3], 
	# 						img[y2][x0], img[y2][x1], img[y2][x2], img[y2][x3], 
	# 						img[y3][x0], img[y3][x1], img[y3][x2], img[y3][x3]]
	# sample255TruthList = [ pix > 5 for pix in pixelValueSampleList]
	# matrix = img if True in sample255TruthList else img * 255
	adjustWeight = 255/(np.mean(img)/0.2)
	matrix = img * adjustWeight

	print('Info: adjustWeight ==', adjustWeight)
	print('      imgX.shape ==', imgX.shape)

	# if True in sample255TruthList and reportImageInfo is True:
	# 	print('Info: imgPainter.autoPaint() is painting %s' % path)
	# 	print('      255 level image, size: %s' % str(imgX.shape))
	# elif reportImageInfo is True:
	# 	print('Info: imgPainter.autoPaint() is painting %s' % path)
	# 	print('      float value image, size: %s' % str(imgX.shape))

	matrix = fixMatrixDataOverflow(matrix = matrix) if fixDataOverflow is True else matrix
	paintMatrix(matrix = matrix, path = path)




'''=============================================================================================================='''
'''=============================================================================================================='''
'''=============================================================================================================='''
'''=============================================================================================================='''


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--fltPath', dest = 'fltPath', default = './*.flt', help = 'file path of a flt file')
	parser.add_argument('--outPath', dest = 'outPath', default = './autoPaint.png', help = 'output file path')
	args = parser.parse_args()

	paintFltFileToPng(fltFilePath = args.fltPath, outputFilePath = args.outPath)



