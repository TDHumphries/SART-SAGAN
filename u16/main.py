

import argparse
import copy
import glob
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.client import device_lib

import dataLoader
import debug
import imgEvaluation
import imgFormatConvert
import imgPainter
import imgTrainset
import NNmodel
import parameters
import sinogram
import reconstruction

DRCT_LOAD_SINO = False

if __name__ == '__main__':

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--version', dest = 'version', type = str, default = 'uiharu v16.0.0213.1525', help = 'version of program')
	parser.add_argument('--function', dest = 'function', type = str, default = 'default_function', help = 'trainNN, autoRecon')
	# for trainNN
	parser.add_argument('--cleanTrainset', dest = 'cleanTrainset', type = str, default = '../NDCT', help = '')
	parser.add_argument('--cleanTrainsetDataType', dest = 'cleanTrainsetDataType', type = str, default = 'png', help = '')
	parser.add_argument('--noisyTrainset', dest = 'noisyTrainset', type = str, default = '../uiharuDataset/sparseView100ns50iter20;../uiharuDataset/low-dose1e5ns50iter20;../uiharuDataset/limitedAngle140ns50iter20;../uiharuDataset/dataset1107_sv50ns25it20/sparse_view_50', help = '')
	parser.add_argument('--noisyTrainsetDataType', dest = 'noisyTrainsetDataType', type = str, default = 'png', help = '')
	parser.add_argument('--cleanTestset', dest = 'cleanTestset', type = str, default = '../NDCTtest', help = '')
	parser.add_argument('--cleanTestsetDataType', dest = 'cleanTestsetDataType', type = str, default = 'png', help = '')
	parser.add_argument('--noisyTestset', dest = 'noisyTestset', type = str, default = '../uiharuDataset/combinedTest', help = '')
	parser.add_argument('--noisyTestsetDataType', dest = 'noisyTestsetDataType', type = str, default = 'png', help = '')
	parser.add_argument('--checkpointFolder', dest = 'checkpointFolder', type = str, default = './checkpoint', help = '')
	parser.add_argument('--batchSize', dest = 'batchSize', type = int, default = 1, help = '')
	parser.add_argument('--NNtype', dest = 'NNtype', type = str, default = 'gan', help = 'gan, cycleGAN, IDGAN, testGAN')
	parser.add_argument('--NNstructure_G', dest = 'NNstructure_G', type = str, default = 'simpleGAN_G', help = '')
	parser.add_argument('--NNstructure_D', dest = 'NNstructure_D', type = str, default = 'simpleGAN_D', help = '')
	# for autoRecon
	parser.add_argument('--inputFolder', dest = 'inputFolder', type = str, default = '../NDCTtest', help = '')
	parser.add_argument('--dataType', dest = 'dataType', type = str, default = 'png', help = 'flt, png')
	parser.add_argument('--sinoFolder', dest = 'sinoFolder', type = str, default = './sinogram', help = '')
	parser.add_argument('--noiseOption', dest = 'noiseOption', type = str, default = 'default_option', help = 'default_option, low-dose, sparse_view_450, limited_angle_160')
	parser.add_argument('--mnnOrder', dest = 'mnnOrder', type = str, default = 'sart5|(gan)./checkpoint|sart20|return')
	parser.add_argument('--outputFolder', dest = 'outputFolder', type = str, default = 'default_option', help = './result')
	parser.add_argument('--maxTestNum', dest = 'maxTestNum', default = None, help = '')
	parser.add_argument('--calFNIQA', dest = 'calFNIQA', type = bool, default = False, help = '')
	parser.add_argument('--denoiserCKPTorder', dest = 'denoiserCKPTorder', type = str, default = '(gan)../temp/ckpt_simpleGAN_withSparseView100', help = '')
	parser.add_argument('--FNIQAckpt', dest = 'FNIQAckpt', type = str, default = '../temp/ckpt_cycleGAN-modifiedDnCNN-changedSimpleGAN_withSparseView100', help = '')
	parser.add_argument('--ns', dest = 'ns', type = int, default = 20, help = '')


	
	# else
	parser.add_argument('--verbose', dest = 'verbose', default = None, help = '')
	parser.add_argument('--argOption', dest = 'argOption', type = str, default = 'default', help = 'simpleGAN_test, cycleGAN_test')
	args = parser.parse_args()

	print('================================================================')
	print('Project', args.version)
	print('2020-02-13 15:25')
	print('')

	parameters.globalParametersInitialization()

	# argOption is used to set parameters automatically
	# for beginner we recomennd set parameters manually

	if args.argOption == 'cycleGAN_test':
		args.function = 'autoRecon'
		args.inputFolder = '../NDCTtest'
		args.dataType = 'png'
		args.sinoFolder = './sinogram'
		args.noiseOption = 'low-dose_1e5'
		zzCycleGAN = '(cycleGAN)../temp/ckpt_cycleGAN-modifiedDnCNN-changedSimpleGAN_withSparseView100|'
		zzSimpleGAN = '(gan)../temp/ckpt_simpleGAN_withLowDose1e5|'
		args.mnnOrder = 'sart30|' + zzCycleGAN + 'sart20|' + zzSimpleGAN + 'sart20|' + zzSimpleGAN + 'sart20|' + zzSimpleGAN + 'sart30|' + zzSimpleGAN + 'sart2|return'
		args.outputFolder = './result'
		# args.maxTestNum = 5 if args.maxTestNum is None else int(args.maxTestNum)
	# elif args.argOption == 'test7b':
	# 	args.function = 'autoRecon'
	# 	args.inputFolder = '../NDCTtest'
	# 	args.dataType = 'png'
	# 	args.sinoFolder = './sinogram'
	# 	args.noiseOption = 'sparse_view_60' if args.noiseOption == 'default_option' else args.noiseOption
	# 	zzDDGAN = '(testGAN)../tempCKPT/ckpt_0707|'
	# 	args.mnnOrder = 'sart30|' + zzDDGAN + 'sart20|' + zzDDGAN + 'sart20|' + zzDDGAN + 'sart20|' + zzDDGAN + 'sart30|' + zzDDGAN + 'return'
	# 	args.outputFolder = './result/0707-SV60' if args.outputFolder == 'default_option' else args.outputFolder
	# 	args.calFNIQA = False
	# 	args.denoiserCKPTorder = zzDDGAN
	elif args.argOption == 'testNN716':
		NNpath = '(testGAN)../tempCKPT/ckpt_NN716|'
		noiseType = 'sparse_view_60'
		args.function = 'autoRecon'
		args.inputFolder = '../NDCTtest'
		args.dataType = 'png'
		args.sinoFolder = './sinogram'
		args.noiseOption = noiseType if args.noiseOption == 'default_option' else args.noiseOption
		args.mnnOrder = 'sart30|' + NNpath + 'sart30|' + NNpath + 'sart30|' + NNpath + 'sart30|' + NNpath + 'return'
		args.outputFolder = './result/NN716-' + args.noiseOption if args.outputFolder == 'default_option' else args.outputFolder
		args.calFNIQA = False
		args.denoiserCKPTorder = NNpath
	elif args.argOption == 'NN721':
		NNpath = '(NN721)../tempCKPT/ckpt_NN721|'
		noiseType = 'limited_angle_140'
		args.noiseOption = noiseType if args.noiseOption == 'default_option' else args.noiseOption
		args.NNtype = 'NN721'
		args.function = 'autoRecon'
		args.inputFolder = '../NDCTtest'
		args.dataType = 'png'
		args.sinoFolder = './sinogram_' + args.noiseOption
		if not os.path.exists(args.sinoFolder): os.mkdir(args.sinoFolder)
		args.noiseOption = noiseType if args.noiseOption == 'default_option' else args.noiseOption
		args.mnnOrder = 'sart10ns30|' + NNpath + 'sart10ns30|' + NNpath + 'sart10ns30|' + NNpath + 'sart10ns30|' + NNpath + 'sart10ns30|' + NNpath + 'sart10ns30|' + NNpath + 'sart10ns30|' + NNpath + 'sart10ns30|' + NNpath + 'return'
		args.outputFolder = './result/NN721-' + args.noiseOption if args.outputFolder == 'default_option' else args.outputFolder
		args.calFNIQA = False
		args.denoiserCKPTorder = NNpath
	elif args.argOption == 'NN721NSA':
		NNpath = '(NN721NSA)../tempCKPT/ckpt_NN721NSA_2|'
		sartCommand = 'sart20ns' + str(args.ns) + '|'
		noiseType = 'limited_angle_140'
		args.noiseOption = noiseType if args.noiseOption == 'default_option' else args.noiseOption
		args.NNtype = 'NN721NSA'
		args.function = 'autoRecon'
		args.inputFolder = '../NDCTtest'
		args.dataType = 'png'
		args.sinoFolder = './sinogram_' + args.noiseOption
		if not os.path.exists(args.sinoFolder): os.mkdir(args.sinoFolder)
		args.mnnOrder = sartCommand + NNpath + sartCommand + NNpath + sartCommand + NNpath + sartCommand + NNpath + 'return'
		args.outputFolder = './result/NN721NSA_2-' + args.noiseOption if args.outputFolder == 'default_option' else args.outputFolder
		args.calFNIQA = False
		args.denoiserCKPTorder = NNpath
	# elif args.argOption == 'NN1107':
	# 	NNpath = '(NN1107)../tempCKPT/ckpt_NN1107|'
	# 	sartCommand = 'sart20ns' + str(args.ns) + '|'
	# 	args.noiseOption = 'limited_angle_140' if args.noiseOption == 'default_option' else args.noiseOption
	# 	args.NNtype = 'NN1107'
	# 	args.function = 'autoRecon'
	# 	args.inputFolder = '../NDCTtest'
	# 	args.dataType = 'png'
	# 	args.sinoFolder = './sinogram_' + args.noiseOption
	# 	if not os.path.exists(args.sinoFolder): os.mkdir(args.sinoFolder)
	# 	args.mnnOrder = sartCommand + NNpath + sartCommand + NNpath + sartCommand + NNpath + sartCommand + NNpath + 'return'
	# 	args.outputFolder = './result/NN721NSA_2-' + args.noiseOption if args.outputFolder == 'default_option' else args.outputFolder
	# 	args.calFNIQA = False
	# 	args.denoiserCKPTorder = NNpath
	# elif args.argOption == 'NN1007':
	# 	NNpath = '(NN1007cycle)../tempCKPT/ckpt_1007cycle|'
	# 	sartCommand = 'sart20ns' + str(args.ns) + '|'
	# 	noiseType = 'limited_angle_140'
	# 	args.NNtype = 'NN1007cycle'
	# 	args.function = 'autoRecon' if args.function == 'default_function' else args.function
	# 	args.inputFolder = '../NDCTtest'
	# 	args.dataType = 'png'
	# 	args.sinoFolder = './sinogram'
	# 	args.noiseOption = noiseType if args.noiseOption == 'default_option' else args.noiseOption
	# 	args.mnnOrder = sartCommand + NNpath + sartCommand + NNpath + sartCommand + NNpath + sartCommand + NNpath + 'return'
	# 	args.outputFolder = './result/NN1007cycle-' + args.noiseOption if args.outputFolder == 'default_option' else args.outputFolder
	# 	args.calFNIQA = False
	# 	args.denoiserCKPTorder = NNpath
	# elif args.argOption == 'NN1007cycleP':
	# 	NNpath = '(NN1007cycleP)../tempCKPT/ckpt_NN1007-up64|'
	# 	sartCommand = 'sart20ns' + str(args.ns) + '|'
	# 	noiseType = 'limited_angle_140'
	# 	args.noiseOption = noiseType if args.noiseOption == 'default_option' else args.noiseOption
	# 	args.NNtype = 'NN1007cycle'
	# 	args.function = 'autoRecon' if args.function == 'default_function' else args.function
	# 	args.inputFolder = '../NDCTtest'
	# 	args.dataType = 'png'
	# 	args.sinoFolder = './sinogram_' + args.noiseOption
	# 	if not os.path.exists(args.sinoFolder): os.mkdir(args.sinoFolder)
	# 	args.mnnOrder = sartCommand + NNpath + sartCommand + NNpath + sartCommand + NNpath + sartCommand + NNpath + 'return'
	# 	args.outputFolder = './result/NN1007cycle-' + args.noiseOption if args.outputFolder == 'default_option' else args.outputFolder
	# 	args.calFNIQA = False
	# 	args.denoiserCKPTorder = NNpath

	args.maxTestNum = len(glob.glob(args.inputFolder + '/*.' + args.dataType)) if args.maxTestNum == None else min(len(glob.glob(args.inputFolder + '/*.' + args.dataType)), int(args.maxTestNum))

	# print(args)

	if args.function == 'trainNN':

		cleanImgLoader = dataLoader.folderPathToImgLoader(floderPath = args.cleanTrainset, dataType = args.cleanTrainsetDataType)
		noisyImgLoaderList = [ dataLoader.folderPathToImgLoader(floderPath = noisyTrainsetPath, dataType = args.noisyTrainsetDataType) for noisyTrainsetPath in args.noisyTrainset.split(';') ]
		trainImgLoader_clean, trainImgLoader_noisy = imgTrainset.createTrainsetPair(cleanImgLoader = cleanImgLoader, noisyImgLoaderList = noisyImgLoaderList)
		evaluationImgLoader_clean = dataLoader.folderPathToImgLoader(floderPath = args.cleanTestset, dataType = args.cleanTestsetDataType)
		evaluationImgLoader_noisy = dataLoader.folderPathToImgLoader(floderPath = args.noisyTestset, dataType = args.noisyTestsetDataType)
		# debug.pairLoaderCheck(imgLoader_1 = trainImgLoader_clean, imgLoader_2 = trainImgLoader_noisy)

		if not os.path.exists(args.checkpointFolder): os.mkdir(args.checkpointFolder)

		sessConfig = tf.ConfigProto()
		sessConfig.gpu_options.allow_growth = True
		sessConfig.gpu_options.per_process_gpu_memory_fraction = 1.0
		with tf.Session(config = sessConfig) as sess:
			if args.NNtype == 'testGAN':
				projectModel = NNmodel.Testmodel(sess = sess, checkpointFolderPath = args.checkpointFolder, resultFolderPath = args.checkpointFolder + '/trainingResult')
			elif args.NNtype == 'DDGAN':
				projectModel = NNmodel.DDGANmodel(sess = sess, checkpointFolderPath = args.checkpointFolder, resultFolderPath = args.checkpointFolder + '/trainingResult')
			elif args.NNtype == 'NN716':
				projectModel = NNmodel.NN716(sess = sess, checkpointFolderPath = args.checkpointFolder, resultFolderPath = args.checkpointFolder + '/trainingResult')
			elif args.NNtype == 'NN721':
				projectModel = NNmodel.NN721(sess = sess, checkpointFolderPath = args.checkpointFolder, resultFolderPath = args.checkpointFolder + '/trainingResult')
			elif args.NNtype == 'NN721NSA':
				projectModel = NNmodel.NN721NSA(sess = sess, checkpointFolderPath = args.checkpointFolder, resultFolderPath = args.checkpointFolder + '/trainingResult', usePatch = True, batchSize = float(args.batchSize))
			# elif args.NNtype == 'NN814':
			# 	projectModel = NNmodel.NN814(sess = sess, checkpointFolderPath = args.checkpointFolder, resultFolderPath = args.checkpointFolder + '/trainingResult')
			# elif args.NNtype == 'NN1007cycle':
			# 	projectModel = NNmodel.NN1007cycle(sess = sess, checkpointFolderPath = args.checkpointFolder, resultFolderPath = args.checkpointFolder + '/trainingResult')
			# elif args.NNtype == 'NN1007cycleP':
			# 	projectModel = NNmodel.NN1007cycleP(sess = sess, checkpointFolderPath = args.checkpointFolder, resultFolderPath = args.checkpointFolder + '/trainingResult')
			# elif args.NNtype == 'NN1107':
			# 	projectModel = NNmodel.NN1107(sess = sess, checkpointFolderPath = args.checkpointFolder, resultFolderPath = args.checkpointFolder + '/trainingResult')
			
			else:
				print('ERROR: Unknow NNtype: %s' % str(args.NNtype))
				print('       FATAL ERROR, FORCE EXIT')
				exit()
			checkpointStatus = projectModel.loadModel()
			projectModel.train(trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean, evaluationImgLoader_noisy)

	elif args.function == 'testNN':
		print('Info: --function=testNN, start')
		cleanImgLoader = dataLoader.folderPathToImgLoader(floderPath = args.cleanTrainset, dataType = args.cleanTrainsetDataType)
		noisyImgLoaderList = [ dataLoader.folderPathToImgLoader(floderPath = noisyTrainsetPath, dataType = args.noisyTrainsetDataType) for noisyTrainsetPath in args.noisyTrainset.split(';') ]
		trainImgLoader_clean, trainImgLoader_noisy = imgTrainset.createTrainsetPair(cleanImgLoader = cleanImgLoader, noisyImgLoaderList = noisyImgLoaderList)
		evaluationImgLoader_clean = dataLoader.folderPathToImgLoader(floderPath = args.cleanTestset, dataType = args.cleanTestsetDataType)
		evaluationImgLoader_noisy = dataLoader.folderPathToImgLoader(floderPath = args.noisyTestset, dataType = args.noisyTestsetDataType)
		# debug.pairLoaderCheck(imgLoader_1 = trainImgLoader_clean, imgLoader_2 = trainImgLoader_noisy)

		if not os.path.exists(args.checkpointFolder):
			print('ERROR: Checkpoint folder [%s] not exist.' % str(args.checkpointFolder))
			print('       FATAL ERROR, FORCE EXIT')
			exit()

		sessConfig = tf.ConfigProto()
		sessConfig.gpu_options.allow_growth = True
		sessConfig.gpu_options.per_process_gpu_memory_fraction = 1.0
		with tf.Session(config = sessConfig) as sess:
			if args.NNtype == 'NN716':
				projectModel = NNmodel.NN716(sess = sess, checkpointFolderPath = args.checkpointFolder, resultFolderPath = args.checkpointFolder + '/trainingResult')
			elif args.NNtype == 'NN721':
				projectModel = NNmodel.NN721(sess = sess, checkpointFolderPath = args.checkpointFolder, resultFolderPath = args.checkpointFolder + '/trainingResult')
			elif args.NNtype == 'NN721NSA':
				projectModel = NNmodel.NN721NSA(sess = sess, checkpointFolderPath = args.checkpointFolder, resultFolderPath = args.checkpointFolder + '/trainingResult')
			else:
				print('ERROR: Unknow NNtype: %s' % str(args.NNtype))
				print('       FATAL ERROR, FORCE EXIT')
				exit()
			checkpointStatus = projectModel.loadModel()
			projectModel.testNN(img = evaluationImgLoader_noisy.loadImg())

	elif args.function == 'convertToSinogram':
		pass
	elif args.function == 'reconstruct':
		pass
	elif args.function == 'autoRecon-TVSART':
		# auto reconstruction with SART and SART-TV only

		if not os.path.exists(args.outputFolder): os.mkdir(args.outputFolder)
		if not os.path.exists(args.outputFolder + '/steps'): os.mkdir(args.outputFolder + '/steps')
		imgLoader = dataLoader.folderPathToImgLoader(floderPath = args.inputFolder, dataType = args.dataType)[:args.maxTestNum]
		_, sinogramImgLoader = sinogram.createSinogram_batch(imgLoader = copy.deepcopy(imgLoader), sinogramFolderPath = args.sinoFolder, option = args.noiseOption)
		_, reconImgLoader_SART = reconstruction.MNNsart_batch(sinogramImgLoader = sinogramImgLoader[:], outputFolderPath = args.outputFolder, option = args.noiseOption, mnnOrder = 'sart59ns'+str(args.ns)+'|return', paintSteps = True, imgMiddleName = '_SART')
		_, reconImgLoader_TVSART = reconstruction.MNNsart_batch(sinogramImgLoader = sinogramImgLoader[:], outputFolderPath = args.outputFolder, option = args.noiseOption, mnnOrder = 'TVsart59ns'+str(args.ns)+'|return', paintSteps = True, imgMiddleName = '_TVSART')
		psnr_SART, psnrList_SART = imgEvaluation.averagePSNR_imgLoader(imgLoader_A = copy.deepcopy(reconImgLoader_SART), imgLoader_B = copy.deepcopy(imgLoader))
		psnr_TVSART, psnrList_TVSART = imgEvaluation.averagePSNR_imgLoader(imgLoader_A = copy.deepcopy(reconImgLoader_TVSART), imgLoader_B = copy.deepcopy(imgLoader))
		ssim_SART, ssimList_SART = imgEvaluation.averageSSIM_imgLoader(imgLoader_A = reconImgLoader_SART[:], imgLoader_B = imgLoader[:])
		ssim_TVSART, ssimList_TVSART = imgEvaluation.averageSSIM_imgLoader(imgLoader_A = reconImgLoader_TVSART[:], imgLoader_B = imgLoader[:])
		print('PSNR-SART   : %s' % str([ '%.2f' % psnr for psnr in psnrList_SART ]))
		print('PSNR-TVSART   : %s' % str([ '%.2f' % psnr for psnr in psnrList_TVSART ]))
		print('SSIM-SART   : %s' % str([ '%.2f' % ssim for ssim in ssimList_SART ]))
		print('SSIM-TVSART : %s' % str([ '%.2f' % ssim for ssim in ssimList_TVSART ]))		
		print('============================================================================================')
		print('        average PSNR-SART    == %s' % str(psnr_SART))
		print('        average PSNR-TVSART  == %s' % str(psnr_TVSART))
		print('        average SSIM-SART    == %s' % str(ssim_SART))
		print('        average SSIM-TVSART  == %s' % str(ssim_TVSART))
		print('============================================================================================')
		
	elif args.function == 'autoRecon':

		if not os.path.exists(args.outputFolder): os.mkdir(args.outputFolder)
		if not os.path.exists(args.outputFolder + '/steps'): os.mkdir(args.outputFolder + '/steps')


		imgLoader = dataLoader.folderPathToImgLoader(floderPath = args.inputFolder, dataType = args.dataType)[:args.maxTestNum]

		if DRCT_LOAD_SINO is True:
			sinogramImgLoader = dataLoader.folderPathToImgLoader(floderPath = './sinogram', dataType = 'flt')
		else:
			# with tf.device([ x.name.replace('device:', '').lower() for x in device_lib.list_local_devices() if 'GPU' in x.name ][-1]):
			_, sinogramImgLoader = sinogram.createSinogram_batch(imgLoader = copy.deepcopy(imgLoader), sinogramFolderPath = args.sinoFolder, option = args.noiseOption)

		# if args.calFNIQA is True:
		# 	print('args.mnnOrder == %s' % args.mnnOrder)
		# 	_, reconImgLoader_MNN, matFList_list = reconstruction.MNNsart_batch(sinogramImgLoader = copy.deepcopy(sinogramImgLoader), outputFolderPath = args.outputFolder, option = args.noiseOption, mnnOrder = args.mnnOrder, paintSteps = True, imgMiddleName = '_MNN', returnMatF = True)
		# else:
		# 	_, reconImgLoader_MNN = reconstruction.MNNsart_batch(sinogramImgLoader = copy.deepcopy(sinogramImgLoader), outputFolderPath = args.outputFolder, option = args.noiseOption, mnnOrder = args.mnnOrder, paintSteps = True, imgMiddleName = '_MNN', returnMatF = False)
		_, reconImgLoader_MNN = reconstruction.MNNsart_batch(sinogramImgLoader = sinogramImgLoader[:], outputFolderPath = args.outputFolder, option = args.noiseOption, mnnOrder = args.mnnOrder, paintSteps = True, imgMiddleName = '_MNN', returnMatF = False)
		# debug.imgLoaderCheck(imgLoader = reconImgLoader_MNN)
		_, reconImgLoader_SART = reconstruction.MNNsart_batch(sinogramImgLoader = sinogramImgLoader[:], outputFolderPath = args.outputFolder, option = args.noiseOption, mnnOrder = 'sart59ns'+str(args.ns)+'|return', paintSteps = True, imgMiddleName = '_SART')
		# debug.imgLoaderCheck(imgLoader = reconImgLoader_SART)
		_, reconImgLoader_SART_NN = reconstruction.MNNsart_batch(sinogramImgLoader = sinogramImgLoader[:], outputFolderPath = args.outputFolder, option = args.noiseOption, mnnOrder = 'sart59ns'+str(args.ns)+'|' + args.denoiserCKPTorder + 'return', paintSteps = True, imgMiddleName = '_SART+NN')
		# debug.imgLoaderCheck(imgLoader = reconImgLoader_SART_NN)
		_, reconImgLoader_TVSART = reconstruction.MNNsart_batch(sinogramImgLoader = sinogramImgLoader[:], outputFolderPath = args.outputFolder, option = args.noiseOption, mnnOrder = 'TVsart59ns'+str(args.ns)+'|return', paintSteps = True, imgMiddleName = '_TVSART')
		# debug.imgLoaderCheck(imgLoader = reconImgLoader_TVSART)

		psnr_MNN, psnrList_MNN = imgEvaluation.averagePSNR_imgLoader(imgLoader_A = copy.deepcopy(reconImgLoader_MNN), imgLoader_B = copy.deepcopy(imgLoader))
		psnr_SART, psnrList_SART = imgEvaluation.averagePSNR_imgLoader(imgLoader_A = copy.deepcopy(reconImgLoader_SART), imgLoader_B = copy.deepcopy(imgLoader))
		psnr_SART_NN, psnrList_SART_NN = imgEvaluation.averagePSNR_imgLoader(imgLoader_A = copy.deepcopy(reconImgLoader_SART_NN), imgLoader_B = copy.deepcopy(imgLoader))
		psnr_TVSART, psnrList_TVSART = imgEvaluation.averagePSNR_imgLoader(imgLoader_A = copy.deepcopy(reconImgLoader_TVSART), imgLoader_B = copy.deepcopy(imgLoader))

		ssim_MNN, ssimList_MNN = imgEvaluation.averageSSIM_imgLoader(imgLoader_A = reconImgLoader_MNN[:], imgLoader_B = imgLoader[:])
		ssim_SART, ssimList_SART = imgEvaluation.averageSSIM_imgLoader(imgLoader_A = reconImgLoader_SART[:], imgLoader_B = imgLoader[:])
		ssim_SART_NN, ssimList_SART_NN = imgEvaluation.averageSSIM_imgLoader(imgLoader_A = reconImgLoader_SART_NN[:], imgLoader_B = imgLoader[:])
		ssim_TVSART, ssimList_TVSART = imgEvaluation.averageSSIM_imgLoader(imgLoader_A = reconImgLoader_TVSART[:], imgLoader_B = imgLoader[:])



		print('============================================================================================')
		# if args.calFNIQA is True:
		# 	_ = imgEvaluation.FNetIQAwithPSNR_DetailedBatch(nnReconMatFList_list = matFList_list, oriReconImgLoader = reconImgLoader_SART[:], oriImgLoader = imgLoader[:], checkpointFolderPath = args.FNIQAckpt, verbose = True)
		# 	print('============================================================================================')
		# 	fniqaList_MNN = imgEvaluation.FNetIQA_imgLoader(nnReconImgLoader = reconImgLoader_MNN[:], oriReconImgLoader = reconImgLoader_SART[:], checkpointFolderPath = args.FNIQAckpt)
		# 	print('FNIQA-MNN   : %s' % str([ '%.2f' % fniqa for fniqa in fniqaList_MNN ]))

		print('PSNR-MNN    : %s' % str([ '%.2f' % psnr for psnr in psnrList_MNN ]))
		print('PSNR-SART   : %s' % str([ '%.2f' % psnr for psnr in psnrList_SART ]))
		print('PSNR-SART+NN: %s' % str([ '%.2f' % psnr for psnr in psnrList_SART_NN ]))
		print('PSNR-TVSART : %s' % str([ '%.2f' % psnr for psnr in psnrList_TVSART ]))

		print('SSIM-MNN    : %s' % str([ '%.2f' % ssim for ssim in ssimList_MNN ]))
		print('SSIM-SART   : %s' % str([ '%.2f' % ssim for ssim in ssimList_SART ]))
		print('SSIM-SART+NN: %s' % str([ '%.2f' % ssim for ssim in ssimList_SART_NN ]))
		print('SSIM-TVSART : %s' % str([ '%.2f' % ssim for ssim in ssimList_TVSART ]))


		print('============================================================================================')
		print('Result: average PSNR-MNN     == %s' % str(psnr_MNN))
		print('        average PSNR-SART    == %s' % str(psnr_SART))
		print('        average PSNR-SART+NN == %s' % str(psnr_SART_NN))
		print('        average PSNR-TVSART  == %s' % str(psnr_TVSART))
		print('')
		print('        average SSIM-MNN     == %s' % str(ssim_MNN))
		print('        average SSIM-SART    == %s' % str(ssim_SART))
		print('        average SSIM-SART+NN == %s' % str(ssim_SART_NN))
		print('        average SSIM-TVSART  == %s' % str(ssim_TVSART))
		print('============================================================================================')


		imgPainter.autoPaint(imgX = reconImgLoader_MNN.loadImg(), path = '../temp/MNNreconTest.png')
		imgPainter.autoPaint(imgX = reconImgLoader_SART.loadImg(), path = '../temp/noNNreconTest.png')
		imgPainter.autoPaint(imgX = reconImgLoader_SART_NN.loadImg(), path = '../temp/singleNNreconTest.png')
		imgPainter.autoPaint(imgX = reconImgLoader_TVSART.loadImg(), path = '../temp/TVSARTreconTest.png')


	else:
		print('ERROR: Unknow --function parameter: %s' % args.function)
		print('       Program exit.')
		exit()
