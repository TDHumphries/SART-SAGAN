
import argparse
import glob
import numpy as np
import tensorflow as tf

import dataLoader
import imgFormatConvert
import NNmodel

def averagePSNR_imgLoader(imgLoader_A, imgLoader_B):
	
	averagePSNR = 0
	batchSize = len(imgLoader_A)
	psnrList = []
	for (ndarray_A, imgPureName_A), (ndarray_B, imgPureName_B) in zip(imgLoader_A, imgLoader_B):
		psnr = calculatePSNR_ndarray(ndarray_A = ndarray_A * 255, ndarray_B = ndarray_B * 255)
		averagePSNR += psnr
		psnrList.append(psnr)

	averagePSNR = averagePSNR / batchSize

	return averagePSNR, psnrList

# def PSNR_filePath(path_A, path_B):
# 	psnr, _ = averagePSNR_imgLoader(imgLoader_A = dataLoader.PngLoader(pngFilePathList = [path_A]), imgLoader_B = dataLoader.PngLoader(pngFilePathList = [path_B]))
# 	return psnr


def calculatePSNR_ndarray(ndarray_A, ndarray_B):
	# assert pixel value range is 0-255 and type is uint8
	mse = ((ndarray_A.astype(np.float) - ndarray_B.astype(np.float)) ** 2).mean()
	psnr = 10 * np.log10(255 ** 2 / mse)
	return psnr


# def FNetIQA_Nto1(NNreconMatFList, oriReconMatF, NNmodelType = 'cycleGAN', G_option = 'modifiedDnCNN_G', D_option = 'changedSimpleGAN_D', checkpointFolderPath = './checkpoint'):

# 	psnrList = []

# 	sessConfig = tf.ConfigProto()
# 	sessConfig.gpu_options.allow_growth = True
# 	sessConfig.gpu_options.per_process_gpu_memory_fraction = 1
# 	with tf.Session(config = sessConfig) as sess:
# 		projectModel = NNmodel.CycleGANmodel(sess = sess, G_option = G_option, D_option = D_option, checkpointFolderPath = checkpointFolderPath)
# 		checkpointStatus = projectModel.loadCheckpoint(checkpointFolderPath = checkpointFolderPath)
# 		NNfunction = lambda x: imgFormatConvert.reshapeImgBatchToMatrix(projectModel.generate_Fnet(imgBatch = imgFormatConvert.reshapeMatrixToImgBatch(imgPainter.fixMatrixDataOverflow(x))))
# 		for NNReconMatF in NNreconMatFList:
# 			matF_afterFnet = NNfunction(NNReconMatF)
# 			# debug.autoImgXCheck(imgX = matF_afterFnet)
# 			psnrList.append(calculatePSNR_ndarray(ndarray_A = matF_afterFnet*255, ndarray_B = oriReconMatF*255))
# 	tf.reset_default_graph()
# 	print('Debug: In imgEvaluation.FNetIQA_Nto1()')
# 	print('       psnrList == %s' % str(psnrList))
# 	return psnrList


def FNetIQA_imgLoader(nnReconImgLoader, oriReconImgLoader, NNmodelType = 'cycleGAN', G_option = 'modifiedDnCNN_G', D_option = 'changedSimpleGAN_D', checkpointFolderPath = './checkpoint'):
	psnrList = []

	sessConfig = tf.ConfigProto()
	sessConfig.gpu_options.allow_growth = True
	sessConfig.gpu_options.per_process_gpu_memory_fraction = 1
	with tf.Session(config = sessConfig) as sess:
		projectModel = NNmodel.CycleGANmodel(sess = sess, G_option = G_option, D_option = D_option, checkpointFolderPath = checkpointFolderPath)
		checkpointStatus = projectModel.loadCheckpoint(checkpointFolderPath = checkpointFolderPath)
		# NNfunction = lambda x: imgFormatConvert.reshapeImgBatchToMatrix(projectModel.generate_Fnet(imgBatch = imgFormatConvert.reshapeMatrixToImgBatch(imgPainter.fixMatrixDataOverflow(x))))
		for ((nnReconImg, nnReconImgPureName), (oriReconImg, oriReconImgPureName)) in zip(nnReconImgLoader, oriReconImgLoader):
			FnetProjectedMatrix = imgFormatConvert.reshapeImgBatchToMatrix(projectModel.generate_Fnet(imgBatch = imgFormatConvert.reshapeImgToImgBatch(nnReconImg)))
			oriReconMatrix = imgFormatConvert.reshapeImgToMatrix(oriReconImg)
			psnrList.append(calculatePSNR_ndarray(ndarray_A = FnetProjectedMatrix*255, ndarray_B = oriReconMatrix*255))
	tf.reset_default_graph()
	# print('Debug: In imgEvaluation.FNetIQA_imgLoader()')
	# print('       psnrList == %s' % str(psnrList))
	return psnrList



def FNetIQAwithPSNR_DetailedBatch(nnReconMatFList_list, oriReconImgLoader, oriImgLoader, NNmodelType = 'cycleGAN', G_option = 'modifiedDnCNN_G', D_option = 'changedSimpleGAN_D', checkpointFolderPath = './checkpoint', verbose = True):

	if len(nnReconMatFList_list) != len(oriReconImgLoader) or len(oriReconImgLoader) != len(oriImgLoader) or len(oriImgLoader) == 0:
		print('ERROR! In imgEvaluation.FNetIQAwithPSNR_DetailedBatch()')
		print('       len(nnReconMatFList_list) == %d' % len(nnReconMatFList_list))
		print('       len(oriReconImgLoader) == %d' % len(oriReconImgLoader))
		print('       len(oriImgLoader) == %d' % len(oriImgLoader))
		return [{'imgPureName': 'ERROR IN imgEvaluation.FNetIQAwithPSNR_DetailedBatch()', 'psnrList': [], 'fniqaList': []}]
	
	evalResultDictList = []

	sessConfig = tf.ConfigProto()
	sessConfig.gpu_options.allow_growth = True
	sessConfig.gpu_options.per_process_gpu_memory_fraction = 1
	with tf.Session(config = sessConfig) as sess:
		projectModel = NNmodel.CycleGANmodel(sess = sess, G_option = G_option, D_option = D_option, checkpointFolderPath = checkpointFolderPath)
		checkpointStatus = projectModel.loadCheckpoint(checkpointFolderPath = checkpointFolderPath)
		NNfunction = lambda x: imgFormatConvert.reshapeImgBatchToMatrix(projectModel.generate_Fnet(imgBatch = imgFormatConvert.reshapeMatrixToImgBatch(imgPainter.fixMatrixDataOverflow(x))))

		for (nnReconMatFList, (oriReconImg, oriReconImgPureName), (oriImg, oriImgPureName)) in zip(nnReconMatFList_list, oriReconImgLoader, oriImgLoader):
			if oriReconImgPureName.split('_')[0] != oriImgPureName.split('_')[0]:
				print('ERROR! In imgEvaluation.FNetIQAwithPSNR_DetailedBatch()')
				print('       oriReconImgPureName == %s' % str(oriReconImgPureName))
				print('       oriImgPureName == %s' % str(oriImgPureName))
				print('       FATAL ERROR, FORCE EXIT')
				exit()

			if verbose is True: print('Info: Detailed Fnet IQA: handling img %s' % str(oriImgPureName))
			psnrList, psnrString = [], 'PSNR:  '
			fniqaList, fniqaString = [], 'FNIQA: '
			for nnReconMatF in nnReconMatFList:
				psnr = calculatePSNR_ndarray(ndarray_A = imgFormatConvert.reshapeMatrixToImg(nnReconMatF)*255, ndarray_B = oriImg*255)
				fniqa = calculatePSNR_ndarray(ndarray_A = imgFormatConvert.reshapeMatrixToImg(nnReconMatF)*255, ndarray_B = oriReconImg*255)
				psnrList.append(psnr)
				fniqaList.append(fniqa)
				psnrString += '%.2f->' % psnr
				fniqaString += '%.2f->' % fniqa
			if verbose is True: print(psnrString)
			if verbose is True: print(fniqaString)
			evalResultDictList.append({'imgPureName': oriImg, 'psnrList': psnrList, 'fniqaList': fniqaList})

	tf.reset_default_graph()
	# print('Debug: In imgEvaluation.FNetIQAwithPSNR_DetailedBatch()')
	# print('       psnrList == %s' % str(psnrList))
	return evalResultDictList

# ====================================================================================================================================================================
# ====================================================================================================================================================================
# ====================================================================================================================================================================
# ====================================================================================================================================================================
# ====================================================================================================================================================================

def calculateSSIM_ndarray(imgA, imgB):
	from skimage.measure import compare_ssim
	return compare_ssim(imgFormatConvert.reshapeImgToMatrix(imgA), imgFormatConvert.reshapeImgToMatrix(imgB))

def averageSSIM_imgLoader(imgLoader_A, imgLoader_B):
	ssimList = []
	if len(imgLoader_A) != len(imgLoader_B) or len(imgLoader_A) == 0 or imgLoader_B == 0:
		print('WARNING! In imgEvaluation.averageSSIM_imgLoader()')
		print('         cannot traverse imgLoader')
		print('         len(imgLoader_A) ==', len(imgLoader_A), '; len(imgLoader_B) ==', len(imgLoader_B))
		print('         FORCE EXIT')
		exit()
		return 0
	for (imgA, imgApureName), (imgB, imgBpureName) in zip(imgLoader_A, imgLoader_B):
		ssimList.append(calculateSSIM_ndarray(imgA, imgB))
	print('SSIM:', ssimList)
	return (sum(ssimList) / len(ssimList)), ssimList




# ====================================================================================================================================================================
# ====================================================================================================================================================================
# ====================================================================================================================================================================
# ====================================================================================================================================================================
# ====================================================================================================================================================================


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--func', dest = 'func', type = str, default = 'default_func')
	parser.add_argument('--inFile', dest = 'inFile', type = str, default = 'default_in')
	parser.add_argument('--inFile2', dest = 'inFile2', type = str, default = '../NDCTtest')
	parser.add_argument('--mid', dest = 'mid', type = str, default = 'default_mid')
	parser.add_argument('--outFile', dest = 'outFile', type = str, default = 'default_out')
	parser.add_argument('--dataType', dest = 'dataType', type = str, default = 'png')
	args = parser.parse_args()

	print('args.func == %s' % str(args.func))
	print('args.inFile == %s' % str(args.inFile))
	print('args.outFile == %s' % str(args.outFile))

	if args.func == 'psnr':
		imgLoader_A = dataLoader.PngLoader(pngFilePathList = glob.glob(args.inFile + '/*.png')) if args.dataType == 'png' else dataLoader.FltLoader(fltFilePathList= glob.glob(args.inFile + '/*.flt')) if args.dataType == 'flt' else None
		imgLoader_B = dataLoader.PngLoader(pngFilePathList = glob.glob(args.inFile + '/*.png')) if args.dataType == 'png' else dataLoader.FltLoader(fltFilePathList= glob.glob(args.inFile + '/*.flt')) if args.dataType == 'flt' else None
		avgPSNR, _ = averagePSNR_imgLoader(imgLoader_A, imgLoader_B)
		print(avgPSNR)

	elif args.func == 'ssim':
		imgLoader_A = dataLoader.PngLoader(pngFilePathList = glob.glob(args.inFile + '/*' + args.mid + '*.png'))
		imgLoader_B = dataLoader.PngLoader(pngFilePathList = glob.glob(args.inFile2 + '/*.png'))
		avgSSIM, _ = averageSSIM_imgLoader(imgLoader_A, imgLoader_B)
		print('average SSIM =', avgSSIM)

	elif args.func == 'cmpPSNR':
		import imgPainter
		import debug
		imgLoader_A = dataLoader.PngLoader(pngFilePathList = [args.inFile])
		imgLoader_B = dataLoader.PngLoader(pngFilePathList = [args.inFile2])
		for ((imgA, imgPureName_A), (imgB, imgPureName_B)) in zip(imgLoader_A, imgLoader_B):

			psnr = calculatePSNR_ndarray(ndarray_A = imgA*255, ndarray_B = imgB*255)
			print('PSNR ori =', psnr)
			debug.detailedImgXCheck(imgX = imgA*255)

			imgA *= 2
			imgB *= 2

			psnr = calculatePSNR_ndarray(ndarray_A = imgA*255, ndarray_B = imgB*255)
			print('PSNR adjusted =', psnr)
			debug.detailedImgXCheck(imgX = imgA*255)


			imgPainter.autoPaint(imgX = imgA, path = '../temp/imgA.png')
			imgPainter.autoPaint(imgX = imgB, path = '../temp/imgB.png')
