
import numpy as np


def reshapeMatrixToImg(matrix):
	return np.reshape(a = matrix, newshape = (matrix.shape[0], matrix.shape[1], 1))

def reshapeImgToMatrix(img):
	return np.reshape(a = img, newshape = (img.shape[0], img.shape[1]))

def reshapeImgToImgBatch(img):
	return np.reshape(a = img, newshape = (1, img.shape[0], img.shape[1], img.shape[2]))

def reshapeImgBatchToImg(imgBatch):
	return imgBatch[0]

def reshapeMatrixToImgBatch(matrix):
	return np.reshape(a = matrix, newshape = (1, matrix.shape[0], matrix.shape[1], 1))

def reshapeImgBatchToMatrix(imgBatch):
	return np.reshape(a = imgBatch, newshape = (imgBatch.shape[1], imgBatch.shape[2]))

def transferImgX_255ToFloat(imgX_255):
	return imgX_255 / 255

def transferImgX_FloatTo255(imgX_Float):
	return imgX_Float * 255
