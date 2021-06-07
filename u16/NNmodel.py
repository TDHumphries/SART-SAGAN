
import abc
import argparse
import copy
import glob
import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.client import device_lib
import time

import dataLoader
import debug
import imgEvaluation
import imgFormatConvert
import imgPainter
import layerBlocks
import NNstructure
import NNstructure_IDGAN
import NNstructure_cycleGAN
import NNstructure_GAN_CIRCLE



'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''




def generatorNN(NNinput, isTraining = True, reuse = tf.AUTO_REUSE, nnOption = None, name = 'G_block', device = '/gpu:0', verbose = True):
	print('Info: In NNmodel.generatorNN() nnOption == %s, name == %s' % (nnOption, name))

	if nnOption == None or nnOption == 'simpleGAN_G' or nnOption == 'simpleGAN':
		return NNstructure.generatorNN_simpleGAN(input = NNinput, is_training = isTraining, output_channels = 1)
	elif nnOption == 'modifiedDnCNN_G' or nnOption == 'modifiedDnCNN':
		return NNstructure.generatorNN_modifiedDnCNN(NNinput = NNinput, isTraining = isTraining, output_channels = 1, name = name, device = device)
	elif nnOption == 'cycleGAN_G' or nnOption == 'cycleGAN':
		return NNstructure.generatorNN_cycleGAN(image = NNinput, reuse = reuse, name = name)
	elif nnOption == 'IDGAN_G' or nnOption == 'IDGAN':
		return NNstructure_IDGAN.generator(input = NNinput)
	elif nnOption == 'SAGAN_G' or nnOption == 'SAGAN':
		return NNstructure_SAGAN.generator()
	elif nnOption == '630_G' or nnOption == '630':
		return NNstructure.generatorNN_GTest(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = 'G_testGenerator', device = device)
	elif nnOption == 'DDGAN_G' or nnOption == 'DDGAN':
		return NNstructure.generatorNN_DDGAN(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = 'G_testGenerator', device = device)
	elif nnOption == '716' or nnOption == '716_G':
		return NNstructure.generatorNN_716(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = 'G_716generator', device = device)
	elif nnOption == '721' or nnOption == '721_G':
		return NNstructure.generatorNN_721(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = 'G_NN721', device = device, verbose = verbose)
	elif nnOption == '721NSA' or nnOption == '721NSA_G':
		return NNstructure.generatorNN_721NSA(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = 'G_NN721', device = device)
	# elif nnOption == '814' or nnOption == '814_G':
	# 	return NNstructure.generatorNN_814(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = 'G_NN814', device = device)
	# elif nnOption == '1007' or nnOption == '1007_G':
	# 	# return NNstructure.generatorNN_simpleGAN(input = NNinput, is_training = isTraining, output_channels = 1)
	# 	return NNstructure_cycleGAN.generatorNN_1007(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = 'G_NN814', device = device)
	# elif nnOption == '1018' or nnOption == '1018_G':
	# 	return NNstructure.generatorNN_1018(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = 'G_NN1018', device = device)
	# elif nnOption == '1107' or nnOption == '1107_G':
	# 	return NNstructure.generatorNN_1107(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = 'G_NN1107', device = device)
	else:
		print('ERROR! In NNmodel.generatorNN()')
		print('       Unknow nnOption: %s' % str(nnOption))
		print('       FATAL ERROR, FORCE EXIT')
		exit()


def discriminatorNN(NNinput, isTraining, reuse = tf.AUTO_REUSE, nnOption = None, name = 'D_block', device = '/gpu:1'):
	print('Info: In NNmodel.discriminatorNN() nnOption == %s, name == %s, reuse == %s' % (nnOption, name, str(reuse)))

	if nnOption == None or nnOption == 'simpleGAN_D':
		return NNstructure.discriminatorNN_simpleGAN(NNinput = NNinput, isTraining = isTraining, reuse = reuse)
	elif nnOption == 'changedSimpleGAN_D' or nnOption == 'changedSimpleGAN':
		return NNstructure.discriminatorNN_changedSimpleGAN(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = name, device = device)
	elif nnOption == 'cycleGAN_D' or nnOption == 'cycleGAN':
		return NNstructure.discriminatorNN_cycleGAN(image = NNinput, reuse = reuse, name = name, device = device)
	elif nnOption == 'srGAN_D' or nnOption == 'srGAN':
		return layerBlocks.discriminator_SRGAN(NNinput = NNinput, isTraining = isTraining)
	elif nnOption == 'IDGAN_D' or nnOption == 'IDGAN':
		return NNstructure_IDGAN.discriminator(input = NNinput, reuse = reuse)
	elif nnOption == 'DTest_D' or nnOption == 'DTest' or nnOption == '630_D' or nnOption == '630' or nnOption == 'DDGAN_D' or nnOption == 'DDGAN':
		return NNstructure.discriminatorNN_DTest(NNinput = NNinput, isTraining = isTraining, reuse = tf.AUTO_REUSE, name = 'D_testDiscriminator', device = device)
	elif nnOption == 'DDGAN_L':
		return NNstructure.discriminatorNN_DDGAN_L(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = name, device = device)
	elif nnOption == '716' or nnOption == '716_D':
		return NNstructure.discriminatorNN_716(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = 'D_716discriminator', device = device)
	elif nnOption == '721' or nnOption == '721_D':
		return NNstructure.discriminatorNN_716(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = 'D_NN721', device = device)
	elif nnOption == '721NSA' or nnOption == '721NSA_D':
		return NNstructure.discriminatorNN_721NSA(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = 'D_NN721', device = device)
	# elif nnOption == '814' or nnOption == '814_D':
	# 	return NNstructure.discriminatorNN_716(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = 'D_NN814', device = device)
	# elif nnOption == '1007' or nnOption == '1007_D':
	# 	return NNstructure_cycleGAN.discriminatorNN_1007(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = 'D_NN1007', device = device)
	# elif nnOption == '1018' or nnOption == '1018_D':
	# 	return NNstructure.discriminatorNN_721NSA(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = 'D_NN1018', device = device)
	# elif nnOption == '1107' or nnOption == '1107_D':
	# 	# return NNstructure.discriminatorNN_716(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = 'D_NN721', device = device)
	# 	return NNstructure.discriminatorNN_1107(NNinput = NNinput, isTraining = isTraining, reuse = reuse, name = 'D_NN1107', device = device)
	else:
		print('ERROR! In NNmodel.discriminatorNN()')
		print('       Unknow nnOption: %s' % str(nnOption))
		print('       FATAL ERROR, FORCE EXIT')
		exit()

def lossFunc(a = None, b = None, X = None, Y = None, Gx = None, DGx = None, Dy = None, LEx = None, LEy = None, Gy = None, batchSize = 1.0, lossOption = None):
	if lossOption == 'l2_loss' or lossOption == None:
		return (1.0 / batchSize) * tf.nn.l2_loss(a - b)
	# elif lossOption == 'IDGAN_loss_D':
	# 	# return -tf.reduce_mean(tf.log(Dy) + tf.log(1.-DGx))
	# 	return tf.nn.l2_loss(Dy - 1.) + tf.nn.l2_loss(DGx - 0.)
	# elif lossOption == 'IDGAN_loss_G':
	# 	return 0.5 * -tf.reduce_mean(tf.log(DGx)) + 1.0 * NNstructure_IDGAN.get_pixel_loss(Y, Gx) + 1.0 * NNstructure_IDGAN.get_smooth_loss(Gx)
	# elif lossOption == 'DDGAN_loss_G':
	# 	return 1.0 * NNstructure_IDGAN.get_pixel_loss(Y, Gx) + 1.0 * NNstructure_IDGAN.get_smooth_loss(Gx)
	# elif lossOption == 'DDGAN_loss_P':
	# 	return 0.5 * NNstructure_IDGAN.get_pixel_loss(Y, Gy) + 0.5 * NNstructure_IDGAN.get_smooth_loss(Gy)
	# elif lossOption == 'DDGAN_loss_D':
	# 	return tf.nn.l2_loss(Dy - 1.) + tf.nn.l2_loss(DGx - 0.)
	# elif lossOption == 'DDGAN_loss_L':
	# 	return 0.5*tf.nn.l2_loss(LEy - 1.) + 0.5*tf.nn.l2_loss(LEx - 0.)
	elif lossOption == 'NN721_loss_G':
		return NNstructure_IDGAN.get_pixel_loss(Y, Gx) + NNstructure_IDGAN.get_smooth_loss(Gx)
	elif lossOption == 'NN721_loss_D':
		return tf.nn.l2_loss(Dy - 1.) + tf.nn.l2_loss(DGx - 0.)
	elif lossOption == 'NN721_loss_DG':
		return tf.nn.l2_loss(DGx - 1.)
		
	else:
		print('ERROR! In NNmodel.lossFunc()')
		print('       Unknow lossOption: %s' % str(lossOption))
		print('       FATAL ERROR, FORCE EXIT')
		exit()

def lossFunc_cycle(A = None, B = None, FGa = None, GFb = None, Gb = None, Fa = None, parameter = None, batchSize = 1.0, lossOption = None):
	if lossOption == 'least_square':
		return NNstructure_cycleGAN.least_square(A = A, B = B)
	elif lossOption == 'cycle_loss':
		return NNstructure_cycleGAN.cycle_loss(A = A, F_GA = FGa, B = B, G_FB = GFb, lambda_ = parameter)
	elif lossOption == 'identity_loss':
		return NNstructure_cycleGAN.identity_loss(A = A, G_B = Gb, B = B, F_A = Fa, gamma = parameter)
	else:
		print('ERROR! In NNmodel.lossFunc_cycle()')
		print('       Unknow lossOption: %s' % str(lossOption))
		print('       FATAL ERROR, FORCE EXIT')
		exit()

def tf_psnr(im1, im2):
	# assert pixel value range is 0-1
	mse = tf.losses.mean_squared_error(labels=im2 * 255.0, predictions=im1 * 255.0)
	# return 1/mse
	return 10.0 * (tf.log(255.0 ** 2 / mse) / tf.log(10.0))



'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''



class NNmodel(metaclass = abc.ABCMeta):

	def __init__(self, sess, modelName = 'default_NN_model', checkpointFolderPath = './checkpoint', resultFolderPath = './result', verbose = True, repeatGPUlist = False):
		self.sess = sess
		self.modelName = modelName
		self.globalStep = 0
		self.checkpointFolderPath = checkpointFolderPath
		self.resultFolderPath = resultFolderPath
		self.verbose = verbose
		self.avaliableGPUlist = [ x.name.replace('device:', '').lower() for x in device_lib.list_local_devices() if 'GPU' in x.name ]
		
		if repeatGPUlist is True:
			self.avaliableGPUlist = self.avaliableGPUlist + self.avaliableGPUlist + self.avaliableGPUlist + self.avaliableGPUlist
		if self.verbose: print('Info: In NNmodel.NNmodel.__init__()')
		if self.verbose: print('      avaliable GPU: %s' % str(self.avaliableGPUlist))

		if not os.path.exists(self.checkpointFolderPath): os.makedirs(self.checkpointFolderPath)
		if not os.path.exists(self.resultFolderPath): os.makedirs(self.resultFolderPath)

	def saveModel(self):
		if self.verbose: print('Info: NNmodel.NNmodel.saveModel(): saving NN model...')
		if not os.path.exists(self.checkpointFolderPath): os.makedirs(self.checkpointFolderPath)
		saver = tf.train.Saver()
		saver.save(self.sess, os.path.join(self.checkpointFolderPath, self.modelName), global_step = self.globalStep)

	def loadModel(self):
		if self.verbose: print('Info: NNmodel.NNmodel.loadModel(): loading NN model...')
		saver = tf.train.Saver()
		ckpt = tf.train.get_checkpoint_state(self.checkpointFolderPath)
		if ckpt and ckpt.model_checkpoint_path:
			full_path = tf.train.latest_checkpoint(self.checkpointFolderPath)
			self.globalStep = int(full_path.split('/')[-1].split('-')[-1])
			saver.restore(self.sess, full_path)
			print('Info: Successfully load model from %s, with global step %d' % (self.checkpointFolderPath, self.globalStep))
			return True
		else:
			self.globalStep = 0
			print('Info: Unable to load model in %s' % self.checkpointFolderPath)
			return False

	def stdEvaluate(self, evaluationImgLoader_clean, evaluationImgLoader_noisy, cleanImgPH, noisyImgPH, generatedImgPH, isTrainingPH, useNN = True, **paraDict):
		if paraDict.get('usePatch', False) is True:
			return self.stdEvaluate_patchVersion(evaluationImgLoader_clean, evaluationImgLoader_noisy, cleanImgPH, noisyImgPH, generatedImgPH, isTrainingPH, useNN)
		if evaluationImgLoader_clean is None or evaluationImgLoader_noisy is None or len(evaluationImgLoader_clean) == 0 or len(evaluationImgLoader_noisy) == 0:
			if self.verbose: print('WARNING! NNmodel.NNmodel.stdEvaluate(): Evaluation imgLoader not provided. Skip evaluation')
			return
		if self.verbose: print('Info: In NNmodel.NNmodel.stdEvaluate(): Start evaluating...')
		psnrList = []
		for ((cleanImg, cleanImgPureName), (noisyImg, noisyImgPureName)) in zip(evaluationImgLoader_clean, evaluationImgLoader_noisy):
			if useNN is True:
				imgXList = self.sess.run( [noisyImgPH, generatedImgPH, cleanImgPH], feed_dict = {noisyImgPH: np.array([noisyImg]).astype(np.float32), cleanImgPH: np.array([cleanImg]).astype(np.float32), isTrainingPH: False})
				imgPainter.autoPaint(imgX = imgPainter.concatImg(imgXList), path = self.resultFolderPath + '/evalResult_%s_%s.png' % (str(cleanImgPureName), str(self.globalStep)), reportImageInfo = False)
			else:
				imgXList = [None, np.array([noisyImg]).astype(np.float32), np.array([cleanImg]).astype(np.float32)]
			psnrList.append(imgEvaluation.calculatePSNR_ndarray(np.clip(255*imgXList[1], 0, 255).astype('uint8'), np.clip(255*imgXList[2], 0, 255).astype('uint8')))
		avg_psnr = sum(psnrList)/len(psnrList) if len(psnrList) != 0 else -1
		# self.psnrRecord.append(avg_psnr)
		print('Info: In NNmodel.GANmodel.evaluate(): globalStep: %d; Average PSNR: %.2f' % (self.globalStep, avg_psnr))
		print('================================================================================')

	def stdEvaluate_patchVersion(self, evaluationImgLoader_clean, evaluationImgLoader_noisy, cleanImgPH, noisyImgPH, generatedImgPH, isTrainingPH, useNN = True, **paraDict):
		if evaluationImgLoader_clean is None or evaluationImgLoader_noisy is None or len(evaluationImgLoader_clean) == 0 or len(evaluationImgLoader_noisy) == 0:
			if self.verbose: print('WARNING! NNmodel.NNmodel.stdEvaluate_patchVersion(): Evaluation imgLoader not provided. Skip evaluation')
			return
		if self.verbose: print('Info: In NNmodel.NNmodel.stdEvaluate_patchVersion(): Start evaluating...')
		psnrList = []
		while len(evaluationImgLoader_clean) > 0 and len(evaluationImgLoader_clean) == len(evaluationImgLoader_noisy):
			(cleanImgBatch, cleanImgPureName), (noisyImgBatch, noisyImgPureName) = evaluationImgLoader_clean.loadImgPatch_withName(sidelength_X = 56, sidelength_Y = 56), evaluationImgLoader_noisy.loadImgPatch_withName(sidelength_X = 56, sidelength_Y = 56)
			imgXList = self.sess.run( [noisyImgPH, generatedImgPH, cleanImgPH], feed_dict = {noisyImgPH: noisyImgBatch, cleanImgPH: cleanImgBatch, isTrainingPH: False})
			outputImgPatchList = imgXList[:]
			imgXList = [np.concatenate(seq = patchedImgX, axis = 0) for patchedImgX in imgXList]
			psnrList.append(imgEvaluation.calculatePSNR_ndarray(np.clip(255*imgXList[1], 0, 255).astype('uint8'), np.clip(255*imgXList[2], 0, 255).astype('uint8')))
			concatImgNumPerLine = int(np.sqrt(outputImgPatchList[0].shape[0]))
			# print('Debug: In NNmodel.NNmodel.stdEvaluate_patchVersion(): concatImgNumPerLine ==', concatImgNumPerLine)
			imgPainter.autoPaint(imgPainter.concatImg([ np.concatenate([ np.concatenate(outputImgPatch[i*concatImgNumPerLine:(i+1)*concatImgNumPerLine], axis = 1) for i in range(concatImgNumPerLine) ], axis = 0) for outputImgPatch in outputImgPatchList ]), path = self.resultFolderPath + '/evalResult_%s_%s.png' % (str(cleanImgPureName), str(self.globalStep)), reportImageInfo = False)
			# zzFinalImgList = []
			# i = 0
			# for pGroup in outputImgPatchList:
			# 	pLines = []
			# 	for i in range(concatImgNumPerLine):
			# 		pInLine = pGroup[i*concatImgNumPerLine:(i+1)*concatImgNumPerLine]
			# 		print('===========================================================')
			# 		print(pInLine)
			# 		print(type(pInLine))
			# 		print(pInLine.shape)
			# 		pInLine = np.array(pInLine)
			# 		print('===========================================================')
			# 		pLine = np.concatenate(pInLine, axis = 1)
			# 		pLines.append(pLine)
			# 	pFinal = np.concatenate(pLines, axis = 0)
			# 	zzFinalImgList.append(pFinal)
			# zzFinalImg = imgPainter.concatImg(zzFinalImgList)
			# zzImgPureName = self.resultFolderPath + '/evalResult_%s_%s.png' % (str(cleanImgPureName), str(self.globalStep))
			# imgPainter.autoPaint(zzFinalImg, path = zzImgPureName, reportImageInfo = False)
		
		avg_psnr = sum(psnrList)/len(psnrList) if len(psnrList) != 0 else -1
		# self.psnrRecord.append(avg_psnr)
		print('Info: In NNmodel.GANmodel.evaluate(): globalStep: %d; Average PSNR: %.2f' % (self.globalStep, avg_psnr))
		print('================================================================================')

	def stdGenerate(self, img, noisyImgPH, generatedImgPH, isTrainingPH, **paraDict):
		if paraDict.get('usePatch', False) is True:
			q = int(np.sqrt(int(noisyImgPH.shape[0])))
			p = int(img.shape[0]/q)
			return self.stdGenerate_patchVersion(np.concatenate([ [ img[i*p:(i+1)*p, j*p:(j+1)*p] for j in range(q) ] for i in range(q) ], axis = 0), noisyImgPH, generatedImgPH, isTrainingPH)
		if self.verbose: print('Info: NNmodel.NNmodel.stdGenerate() start.')
		# debug.autoImgXCheck(imgX = imgBatch, info = 'in GANmodel.generate(), imgBatch before NN, should be float')
		imgBatch = imgFormatConvert.reshapeImgToImgBatch(img).astype(np.float32)
		outputImgBatchList = self.sess.run([generatedImgPH], feed_dict = {noisyImgPH: imgBatch, isTrainingPH: False})
		# debug.autoImgXCheck(imgX = outputImgBatchList[0], info = 'in GANmodel.generate(), outputImgBatch after NN, should be float')
		return imgFormatConvert.reshapeImgBatchToImg(outputImgBatchList[0])

	def stdGenerate_patchVersion(self, imgPatch, noisyImgPH, generatedImgPH, isTrainingPH, **paraDict):
		if self.verbose: print('Info: NNmodel.NNmodel.stdGenerate_patchVersion() start.')
		outputImgBatchList = self.sess.run([generatedImgPH], feed_dict = {noisyImgPH: imgPatch, isTrainingPH: False})
		outputImgPatch = outputImgBatchList[0]
		concatImgNumPerLine = int(np.sqrt(outputImgPatch.shape[0]))
		return np.concatenate([ np.concatenate(outputImgPatch[i*concatImgNumPerLine:(i+1)*concatImgNumPerLine], axis = 1) for i in range(concatImgNumPerLine) ], axis = 0)


	def stdTrain(self, trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean, evaluationImgLoader_noisy, epochNumPH, batchSizePH, learningRatePH, learningRateListPH, cleanImgPH, noisyImgPH, generatedImgPH, isTrainingPH, sessRunParaDictList = [{'name':['D_solver', 'D_loss'], 'para':[None, None]}, {'name':['G_solver', 'G_loss'], 'para':[None, None]}], **paraDict):
		if self.verbose: print('Info: NNmodel.IDGANmodel.train() start.')

		if paraDict.get('usePatch', False) is True:
			print('Debug: In NNmodel.NNmodel.stdTrain(): usePatch == True')
		else:
			print('Debug: In NNmodel.NNmodel.stdTrain(): usePatch == False')

		self.stdEvaluate(evaluationImgLoader_clean = evaluationImgLoader_clean[:], evaluationImgLoader_noisy = evaluationImgLoader_noisy[:], cleanImgPH = cleanImgPH, noisyImgPH = noisyImgPH, generatedImgPH = generatedImgPH, isTrainingPH = isTrainingPH, useNN = False)
		# print('Debug: In NNmodel.NNmode.stdTrain(): line 270')
		# self.stdEvaluate(usePatch = True, useNN = True, evaluationImgLoader_clean = evaluationImgLoader_clean[:], evaluationImgLoader_noisy = evaluationImgLoader_noisy[:], cleanImgPH = cleanImgPH, noisyImgPH = noisyImgPH, generatedImgPH = generatedImgPH, isTrainingPH = isTrainingPH)

		batchNum = int(len(trainImgLoader_clean) / batchSizePH)
		startEpoch = int(self.globalStep / batchNum)
		startTime = time.time()
		print('Info: Start training, from epoch %d: ' % startEpoch)

		for epoch in range(startEpoch, epochNumPH):
			print('Info: Epoch %s of %s' % (str(epoch), str(epochNumPH)))
			cleanImgLoader = trainImgLoader_clean[:]
			noisyImgLoader = trainImgLoader_noisy[:]

			for currentStep in range(batchNum):
				self.globalStep += 1
				if paraDict.get('usePatch', False) is True:
					imgBatch_clean = cleanImgLoader.loadImgPatch(sidelength_X = paraDict.get('patchSideLength_X', 128), sidelength_Y = paraDict.get('patchSideLength_Y', 128), batchSize = int(batchSizePH)).astype(np.float32)
					imgBatch_noisy = noisyImgLoader.loadImgPatch(sidelength_X = paraDict.get('patchSideLength_X', 128), sidelength_Y = paraDict.get('patchSideLength_Y', 128), batchSize = int(batchSizePH)).astype(np.float32)
				else:
					imgBatch_clean = cleanImgLoader.loadImgBatch(batchSize = int(batchSizePH)).astype(np.float32)
					imgBatch_noisy = noisyImgLoader.loadImgBatch(batchSize = int(batchSizePH)).astype(np.float32)

				if currentStep % 10 == 0: reportStr = ''
				for sessRunParaDict in sessRunParaDictList:
					sessRunResultList = self.sess.run( sessRunParaDict['para'],
													feed_dict = {noisyImgPH: imgBatch_noisy, cleanImgPH: imgBatch_clean, learningRatePH: learningRateListPH[epoch], isTrainingPH: True})
					# print('Debug: In NNmodel.NNmodel.stdTrain(): line 245: sessRunResultList[1] =', sessRunResultList[1])
					if currentStep % 10 == 0: reportStr += ''.join([ ', '+paraName+': %.2f' % paraValue for (paraName, paraValue) in zip(sessRunParaDict['name'], sessRunResultList) if ('solver' not in paraName) ])

				if currentStep % 10 == 0: print('Epoch: [%d] Step: [%d/%d] time: %.1f%s' % (epoch, currentStep, batchNum, time.time()-startTime, reportStr))

				if self.globalStep % 100 == 0:
					self.stdEvaluate(evaluationImgLoader_clean = evaluationImgLoader_clean[:], evaluationImgLoader_noisy = evaluationImgLoader_noisy[:], cleanImgPH = cleanImgPH, noisyImgPH = noisyImgPH, generatedImgPH = generatedImgPH, isTrainingPH = isTrainingPH, usePatch = paraDict.get('usePatch', False))
				if self.globalStep % 500 == 0:
					print('Info: Saving model...')
					self.saveModel()
				if self.globalStep == 20:
					print('============================================================')
					print('Info: step 20 model saving test.')
					self.saveModel()
					# self.stdEvaluate(evaluationImgLoader_clean = evaluationImgLoader_clean[:], evaluationImgLoader_noisy = evaluationImgLoader_noisy[:], cleanImgPH = cleanImgPH, noisyImgPH = noisyImgPH, generatedImgPH = generatedImgPH, isTrainingPH = isTrainingPH, usePatch = paraDict.get('usePatch', False))
					print('Info: step 20 model saved.')
					print('============================================================')
			self.stdEvaluate(evaluationImgLoader_clean = evaluationImgLoader_clean[:], evaluationImgLoader_noisy = evaluationImgLoader_noisy[:], cleanImgPH = cleanImgPH, noisyImgPH = noisyImgPH, generatedImgPH = generatedImgPH, isTrainingPH = isTrainingPH, usePatch = paraDict.get('usePatch', False) )
			self.saveModel()

		print('Info: Training finish.')



	@abc.abstractmethod
	def generate(self, img, **paraDict):
		pass

	@abc.abstractmethod
	def train(self, trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean, evaluationImgLoader_noisy, batchSize, **paraDict):
		pass

	@abc.abstractmethod
	def evaluate(self, evaluationImgLoader_clean, evaluationImgLoader_noisy, **paraDict):
		pass
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''



class Testmodel(NNmodel):

	def __init__(self, sess, G_option = '630_G', D_option = '630_D', lossOption = 'IDGAN_loss', **paraDict):
		super().__init__(sess = sess, modelName = 'IDGANmodel-IDGAN', checkpointFolderPath = paraDict.get('checkpointFolderPath', './checkpoint'), resultFolderPath = paraDict.get('resultFolderPath', './result'), verbose = paraDict.get('verbose', True))


		self.globalStep = 0
		self.batchSize = paraDict.get('batchSize', 1.0)
		self.epochNum = 50
		self.learningRateList = 0.0001 * np.ones(self.epochNum)
		self.learningRateList[8:] = self.learningRateList[0]/5.0
		self.learningRateList[12:] = self.learningRateList[0]/10.0
		self.learningRateList[16:] = self.learningRateList[0]/20.0
		self.psnrRecord = []

		# build model -- define placeholder
		self.X = tf.placeholder(tf.float32, [self.batchSize, 512, 512, 1], name = 'low_dose_image')
		self.Y = tf.placeholder(tf.float32, [self.batchSize, 512, 512, 1], name = 'original_image')
		self.isTraining = tf.placeholder(tf.bool, name = 'is_training')
		self.learningRate = tf.placeholder(tf.float32, name = 'learning_rate')
		# build model -- build NN
		self.Gx = generatorNN(NNinput = self.X, nnOption = G_option, name = 'G_block', isTraining = self.isTraining, reuse = False, device = self.avaliableGPUlist[:])
		self.Dy = discriminatorNN(NNinput = self.Y, nnOption = D_option, name = 'D_block', isTraining = self.isTraining, reuse = False, device = self.avaliableGPUlist[0])
		self.DGx = discriminatorNN(NNinput = self.Gx, nnOption = D_option, name = 'D_block', isTraining = self.isTraining, reuse = True, device = self.avaliableGPUlist[0])
		# build model -- create loss
		self.G_loss = lossFunc(X = self.X, DGx = self.DGx, Gx = self.Gx, Y = self.Y, Dy = self.Dy, batchSize = self.batchSize, lossOption = lossOption + '_G')
		self.D_loss = lossFunc(X = self.X, DGx = self.DGx, Gx = self.Gx, Y = self.Y, Dy = self.Dy, batchSize = self.batchSize, lossOption = lossOption + '_D')
		self.psnr = tf_psnr(self.Gx, self.Y)
		self.DTest_DGx = self.DGx
		self.DTest_Dy = self.Dy

		t_vars = tf.trainable_variables()
		d_vars = [var for var in t_vars if 'D_' in var.name]
		g_vars = [var for var in t_vars if 'G_' in var.name]

		with tf.device(self.avaliableGPUlist[0]):
			self.D_solver = tf.train.AdamOptimizer(self.learningRate).minimize(self.D_loss, var_list = d_vars)
			self.G_solver = tf.train.AdamOptimizer(self.learningRate).minimize(self.G_loss+self.D_loss, var_list = g_vars)

		self.sess.run(tf.global_variables_initializer())
		print('Info: Successfully init Testmodel(NNmodel)')

	def train(self, trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean = None, evaluationImgLoader_noisy = None, **paraDict):
		self.stdTrain(trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean, evaluationImgLoader_noisy, \
						epochNumPH = 1, batchSizePH = self.batchSize, learningRatePH = self.learningRate, learningRateListPH = self.learningRateList, \
						cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining, \
						sessRunParaDictList = [	{'name':['D_solver', 'D_loss', 'DGx', 'Dy'], 'para':[self.D_solver, self.D_loss, self.DTest_DGx, self.DTest_Dy]}, \
												{'name':['G_solver', 'G_loss'], 'para':[self.G_solver, self.G_loss]} ], \
						**paraDict)
		self.stdTrain(trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean, evaluationImgLoader_noisy, \
						epochNumPH = 1, batchSizePH = self.batchSize, learningRatePH = self.learningRate, learningRateListPH = self.learningRateList, \
						cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining, \
						sessRunParaDictList = [	{'name':['D_solver', 'D_loss', 'DGx', 'Dy'], 'para':[self.D_solver, self.D_loss, self.DTest_DGx, self.DTest_Dy]} ], \
						**paraDict)
		self.stdTrain(trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean, evaluationImgLoader_noisy, \
						epochNumPH = self.epochNum, batchSizePH = self.batchSize, learningRatePH = self.learningRate, learningRateListPH = self.learningRateList, \
						cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining, \
						sessRunParaDictList = [	{'name':['D_solver', 'D_loss', 'DGx', 'Dy'], 'para':[self.D_solver, self.D_loss, self.DTest_DGx, self.DTest_Dy]}, \
												{'name':['G_solver', 'G_loss'], 'para':[self.G_solver, self.G_loss]} ], \
						**paraDict)

	def evaluate(self, evaluationImgLoader_clean, evaluationImgLoader_noisy, **paraDict):
		self.stdEvaluate(evaluationImgLoader_clean = evaluationImgLoader_clean[:], evaluationImgLoader_noisy = evaluationImgLoader_noisy[:], cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining)
	
	def generate(self, img, **paraDict):
		return self.stdGenerate(img = img, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining)




'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''



class DDGANmodel(NNmodel):

	def __init__(self, sess, G_option = 'DDGAN_G', D_option = '630_D', L_option = 'DDGAN_L', lossOption = 'DDGAN_loss', **paraDict):
		super().__init__(sess = sess, modelName = 'IDGANmodel-IDGAN', checkpointFolderPath = paraDict.get('checkpointFolderPath', './checkpoint'), resultFolderPath = paraDict.get('resultFolderPath', './result'), verbose = paraDict.get('verbose', True))

		self.cutNumber = 1.0

		self.globalStep = 0
		self.batchSize = paraDict.get('batchSize', self.cutNumber*self.cutNumber)
		self.epochNum = 50
		self.learningRateList = 0.0001 * np.ones(self.epochNum)
		self.learningRateList[8:] = self.learningRateList[0]/5.0
		self.learningRateList[12:] = self.learningRateList[0]/10.0
		self.learningRateList[16:] = self.learningRateList[0]/20.0
		self.psnrRecord = []

		# build model -- define placeholder
		self.X = tf.placeholder(tf.float32, [self.batchSize, int(512/self.cutNumber), int(512/self.cutNumber), 1], name = 'low_dose_image')
		self.Y = tf.placeholder(tf.float32, [self.batchSize, int(512/self.cutNumber), int(512/self.cutNumber), 1], name = 'original_image')
		self.isTraining = tf.placeholder(tf.bool, name = 'is_training')
		self.learningRate = tf.placeholder(tf.float32, name = 'learning_rate')
		# build model -- build NN
		self.Gx, self.E0x, self.E1x, self.E2x, self.E3x = generatorNN(NNinput = self.X, nnOption = G_option, name = 'G_block', isTraining = self.isTraining, reuse = False, device = self.avaliableGPUlist[:])
		self.Gy, self.E0y, self.E1y, self.E2y, self.E3y = generatorNN(NNinput = self.Y, nnOption = G_option, name = 'G_block', isTraining = self.isTraining, reuse = True, device = self.avaliableGPUlist[:])
		self.Dy = discriminatorNN(NNinput = self.Y, nnOption = D_option, name = 'D_block', isTraining = self.isTraining, reuse = False, device = self.avaliableGPUlist[0])
		self.DGx = discriminatorNN(NNinput = self.Gx, nnOption = D_option, name = 'D_block', isTraining = self.isTraining, reuse = True, device = self.avaliableGPUlist[0])
		self.LE2x = discriminatorNN(NNinput = self.E2x, nnOption = L_option, name = 'L_block', isTraining = self.isTraining, reuse = False, device = self.avaliableGPUlist[0])
		self.LE2y = discriminatorNN(NNinput = self.E2y, nnOption = L_option, name = 'L_block', isTraining = self.isTraining, reuse = True, device = self.avaliableGPUlist[0])
		# build model -- create loss
		self.G_loss = lossFunc(X = self.X, DGx = self.DGx, Gx = self.Gx, Y = self.Y, Dy = self.Dy, Gy = self.Gy, LEx = self.LE2x, LEy = self.LE2y, batchSize = self.batchSize, lossOption = lossOption + '_G')
		self.D_loss = lossFunc(X = self.X, DGx = self.DGx, Gx = self.Gx, Y = self.Y, Dy = self.Dy, Gy = self.Gy, LEx = self.LE2x, LEy = self.LE2y, batchSize = self.batchSize, lossOption = lossOption + '_D')
		self.P_loss = lossFunc(X = self.X, DGx = self.DGx, Gx = self.Gx, Y = self.Y, Dy = self.Dy, Gy = self.Gy, LEx = self.LE2x, LEy = self.LE2y, batchSize = self.batchSize, lossOption = lossOption + '_P')
		self.L_loss = lossFunc(X = self.X, DGx = self.DGx, Gx = self.Gx, Y = self.Y, Dy = self.Dy, Gy = self.Gy, LEx = self.LE2x, LEy = self.LE2y, batchSize = self.batchSize, lossOption = lossOption + '_L')
		self.psnr = tf_psnr(self.Gx, self.Y)

		t_vars = tf.trainable_variables()
		d_vars = [ var for var in t_vars if 'D_' in var.name ]
		g_vars = [ var for var in t_vars if 'G_' in var.name ]
		l_vars = [ var for var in t_vars if 'L_' in var.name ]
		dl_vars = [ var for var in l_vars if var not in d_vars ] + d_vars

		with tf.device(self.avaliableGPUlist[0]):
			self.DL_solver = tf.train.AdamOptimizer(self.learningRate).minimize(self.D_loss+self.L_loss, var_list = dl_vars)
			self.G_solver = tf.train.AdamOptimizer(self.learningRate).minimize(self.G_loss+self.P_loss+self.D_loss+self.L_loss, var_list = g_vars)

		self.sess.run(tf.global_variables_initializer())
		print('Info: Successfully init Testmodel(NNmodel)')

	def train(self, trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean = None, evaluationImgLoader_noisy = None, **paraDict):
		self.stdTrain(trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean, evaluationImgLoader_noisy, \
						epochNumPH = 1, batchSizePH = self.batchSize, learningRatePH = self.learningRate, learningRateListPH = self.learningRateList, \
						cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining, \
						sessRunParaDictList = [	{'name':['DL_solver', 'D_loss', 'L_loss', 'DGx', 'Dy', 'LE2x', 'LE2y'], 'para':[self.DL_solver, self.D_loss, self.L_loss, self.DGx, self.Dy, self.LE2x, self.LE2y]}, \
												{'name':['G_solver', 'G_loss', 'P_loss'], 'para':[self.G_solver, self.G_loss, self.P_loss]} ], \
						**paraDict)
		self.stdTrain(trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean, evaluationImgLoader_noisy, \
						epochNumPH = 1, batchSizePH = self.batchSize, learningRatePH = self.learningRate, learningRateListPH = self.learningRateList, \
						cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining, \
						sessRunParaDictList = [	{'name':['DL_solver', 'D_loss', 'L_loss', 'DGx', 'Dy', 'LE2x', 'LE2y'], 'para':[self.DL_solver, self.D_loss, self.L_loss, self.DGx, self.Dy, self.LE2x, self.LE2y]} ], \
						**paraDict)
		self.stdTrain(trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean, evaluationImgLoader_noisy, \
						epochNumPH = self.epochNum, batchSizePH = self.batchSize, learningRatePH = self.learningRate, learningRateListPH = self.learningRateList, \
						cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining, \
						sessRunParaDictList = [	{'name':['DL_solver', 'D_loss', 'L_loss', 'DGx', 'Dy', 'LE2x', 'LE2y'], 'para':[self.DL_solver, self.D_loss, self.L_loss, self.DGx, self.Dy, self.LE2x, self.LE2y]}, \
												{'name':['G_solver', 'G_loss', 'P_loss'], 'para':[self.G_solver, self.G_loss, self.P_loss]} ], \
						**paraDict)

	def evaluate(self, evaluationImgLoader_clean, evaluationImgLoader_noisy, **paraDict):
		self.stdEvaluate(evaluationImgLoader_clean = evaluationImgLoader_clean[:], evaluationImgLoader_noisy = evaluationImgLoader_noisy[:], cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining)
	
	def generate(self, img, **paraDict):
		return self.stdGenerate(img = img, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining)

	def testNN(self, img, **paraDict):
		if self.verbose: print('Info: NNmodel.DDGAN.testNN() start.')
		# debug.autoImgXCheck(imgX = imgBatch, info = 'in GANmodel.generate(), imgBatch before NN, should be float')
		imgBatch = imgFormatConvert.reshapeImgToImgBatch(img).astype(np.float32)
		GxImgBatch, E1xImgBatch, E2xImgBatch, E3xImgBatch = self.sess.run([self.Gx, self.E1x, self.E2x, self.E3x], feed_dict = {self.X: imgBatch, self.isTraining: False})
		# GxImgBatch, E1xImgBatch, E2xImgBatch, E3xImgBatch = self.sess.run([self.Gx, self.E1x, self.E2x, self.E3x], feed_dict = {self.X: imgBatch, self.isTraining: False})
		# debug.autoImgXCheck(imgX = outputImgBatchList[0], info = 'in GANmodel.generate(), outputImgBatch after NN, should be float')

		imgPainter.autoPaint(GxImgBatch, path = './temp/Gx.jpg', reportImageInfo = True, fixDataOverflow = True)
		imgPainter.autoPaintPlus(E1xImgBatch, path = './temp/E1x_0.jpg', channel = 0, reportImageInfo = True, fixDataOverflow = True)
		imgPainter.autoPaintPlus(E2xImgBatch, path = './temp/E2x_0.jpg', channel = 0, reportImageInfo = True, fixDataOverflow = True)
		imgPainter.autoPaintPlus(E3xImgBatch, path = './temp/E3x_0.jpg', channel = 0, reportImageInfo = True, fixDataOverflow = False)
		return



'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''


class NN716(NNmodel):

	def __init__(self, sess, G_option = '716_G', D_option = '716_D', lossOption = 'IDGAN_loss', **paraDict):
		super().__init__(sess = sess, modelName = 'NNmodel716', checkpointFolderPath = paraDict.get('checkpointFolderPath', './checkpoint'), resultFolderPath = paraDict.get('resultFolderPath', './result'), verbose = paraDict.get('verbose', True))


		self.globalStep = 0
		self.batchSize = paraDict.get('batchSize', 1.0)
		self.epochNum = 50
		self.learningRateList = 0.0001 * np.ones(self.epochNum)
		self.learningRateList[8:] = self.learningRateList[0]/5.0
		self.learningRateList[12:] = self.learningRateList[0]/10.0
		self.learningRateList[16:] = self.learningRateList[0]/20.0
		self.psnrRecord = []

		# build model -- define placeholder
		self.X = tf.placeholder(tf.float32, [self.batchSize, 512, 512, 1], name = 'low_dose_image')
		self.Y = tf.placeholder(tf.float32, [self.batchSize, 512, 512, 1], name = 'original_image')
		self.isTraining = tf.placeholder(tf.bool, name = 'is_training')
		self.learningRate = tf.placeholder(tf.float32, name = 'learning_rate')
		# build model -- build NN
		self.Gx, self.E0x, self.E1x, self.E2x, self.E3x, self.E4x = generatorNN(NNinput = self.X, nnOption = G_option, name = 'G_block', isTraining = self.isTraining, reuse = False, device = self.avaliableGPUlist[:], verbose = verbose)
		self.Dy = discriminatorNN(NNinput = self.Y, nnOption = D_option, name = 'D_block', isTraining = self.isTraining, reuse = False, device = self.avaliableGPUlist[0])
		self.DGx = discriminatorNN(NNinput = self.Gx, nnOption = D_option, name = 'D_block', isTraining = self.isTraining, reuse = True, device = self.avaliableGPUlist[0])
		# build model -- create loss
		self.G_loss = lossFunc(X = self.X, DGx = self.DGx, Gx = self.Gx, Y = self.Y, Dy = self.Dy, batchSize = self.batchSize, lossOption = lossOption + '_G')
		self.D_loss = lossFunc(X = self.X, DGx = self.DGx, Gx = self.Gx, Y = self.Y, Dy = self.Dy, batchSize = self.batchSize, lossOption = lossOption + '_D')
		self.psnr = tf_psnr(self.Gx, self.Y)
		self.DTest_DGx = self.DGx
		self.DTest_Dy = self.Dy

		t_vars = tf.trainable_variables()
		d_vars = [var for var in t_vars if 'D_' in var.name]
		g_vars = [var for var in t_vars if 'G_' in var.name]

		with tf.device(self.avaliableGPUlist[0]):
			self.D_solver = tf.train.AdamOptimizer(self.learningRate).minimize(self.D_loss, var_list = d_vars)
			self.G_solver = tf.train.AdamOptimizer(self.learningRate).minimize(self.G_loss+self.D_loss, var_list = g_vars)

		self.sess.run(tf.global_variables_initializer())
		print('Info: Successfully init Testmodel(NNmodel)')

	def train(self, trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean = None, evaluationImgLoader_noisy = None, **paraDict):
		self.stdTrain(trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean, evaluationImgLoader_noisy, \
						epochNumPH = 1, batchSizePH = self.batchSize, learningRatePH = self.learningRate, learningRateListPH = self.learningRateList, \
						cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining, \
						sessRunParaDictList = [	{'name':['D_solver', 'D_loss', 'DGx', 'Dy'], 'para':[self.D_solver, self.D_loss, self.DTest_DGx, self.DTest_Dy]}, \
												{'name':['G_solver', 'G_loss'], 'para':[self.G_solver, self.G_loss]} ], \
						**paraDict)
		self.stdTrain(trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean, evaluationImgLoader_noisy, \
						epochNumPH = 1, batchSizePH = self.batchSize, learningRatePH = self.learningRate, learningRateListPH = self.learningRateList, \
						cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining, \
						sessRunParaDictList = [	{'name':['D_solver', 'D_loss', 'DGx', 'Dy'], 'para':[self.D_solver, self.D_loss, self.DTest_DGx, self.DTest_Dy]} ], \
						**paraDict)
		self.stdTrain(trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean, evaluationImgLoader_noisy, \
						epochNumPH = self.epochNum, batchSizePH = self.batchSize, learningRatePH = self.learningRate, learningRateListPH = self.learningRateList, \
						cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining, \
						sessRunParaDictList = [	{'name':['D_solver', 'D_loss', 'DGx', 'Dy'], 'para':[self.D_solver, self.D_loss, self.DTest_DGx, self.DTest_Dy]}, \
												{'name':['G_solver', 'G_loss'], 'para':[self.G_solver, self.G_loss]} ], \
						**paraDict)

	def evaluate(self, evaluationImgLoader_clean, evaluationImgLoader_noisy, **paraDict):
		self.stdEvaluate(evaluationImgLoader_clean = evaluationImgLoader_clean[:], evaluationImgLoader_noisy = evaluationImgLoader_noisy[:], cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining)
	
	def generate(self, img, **paraDict):
		return self.stdGenerate(img = img, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining)

	def testNN(self, img, **paraDict):
		if self.verbose: print('Info: NNmodel.NN716.testNN() start.')
		# debug.autoImgXCheck(imgX = imgBatch, info = 'in GANmodel.generate(), imgBatch before NN, should be float')
		imgBatch = imgFormatConvert.reshapeImgToImgBatch(img).astype(np.float32)
		XImgBatch, GxImgBatch, E0xImgBatch, E1xImgBatch, E2xImgBatch, E3xImgBatch, E4xImgBatch = self.sess.run([self.X, self.Gx, self.E0x, self.E1x, self.E2x, self.E3x, self.E4x], feed_dict = {self.X: imgBatch, self.isTraining: False})
		# GxImgBatch, E1xImgBatch, E2xImgBatch, E3xImgBatch = self.sess.run([self.Gx, self.E1x, self.E2x, self.E3x], feed_dict = {self.X: imgBatch, self.isTraining: False})
		# debug.autoImgXCheck(imgX = outputImgBatchList[0], info = 'in GANmodel.generate(), outputImgBatch after NN, should be float')

		imgPainter.autoPaint(GxImgBatch, path = './temp/Gx.jpg', reportImageInfo = True, fixDataOverflow = True)
		imgPainter.autoPaint(XImgBatch, path = './temp/X.jpg', reportImageInfo = True, fixDataOverflow = True)
		imgPainter.autoPaintPlus(E0xImgBatch, path = './temp/E0x_0.jpg', channel = 0, reportImageInfo = True, fixDataOverflow = True)
		imgPainter.autoPaintPlus(E1xImgBatch, path = './temp/E1x_0.jpg', channel = 0, reportImageInfo = True, fixDataOverflow = True)
		imgPainter.autoPaintPlus(E2xImgBatch, path = './temp/E2x_0.jpg', channel = 0, reportImageInfo = True, fixDataOverflow = True)
		imgPainter.autoPaintPlus(E2xImgBatch, path = './temp/E2x_1.jpg', channel = 1, reportImageInfo = True, fixDataOverflow = False)
		imgPainter.autoPaintPlus(E2xImgBatch, path = './temp/E2x_2.jpg', channel = 2, reportImageInfo = True, fixDataOverflow = False)
		imgPainter.autoPaintPlus(E2xImgBatch, path = './temp/E2x_3.jpg', channel = 3, reportImageInfo = True, fixDataOverflow = False)
		imgPainter.autoPaintPlus(E3xImgBatch, path = './temp/E3x_0.jpg', channel = 0, reportImageInfo = True, fixDataOverflow = False)
		imgPainter.autoPaintPlus(E4xImgBatch, path = './temp/E4x_0.jpg', channel = 0, reportImageInfo = True, fixDataOverflow = False)
		return







'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''


class NN721(NNmodel):
	# std self-attention convlutional auto encoder GAN model

	def __init__(self, sess, G_option = '721_G', D_option = '721_D', lossOption = 'NN721_loss', **paraDict):
		super().__init__(sess = sess, modelName = 'NNmodel721', checkpointFolderPath = paraDict.get('checkpointFolderPath', './checkpoint'), resultFolderPath = paraDict.get('resultFolderPath', './result'), verbose = paraDict.get('verbose', True))

		self.globalStep = 0
		self.batchSize = paraDict.get('batchSize', 1.0)
		self.epochNum = 50
		self.learningRateList = 0.0001 * np.ones(self.epochNum)
		
		self.learningRateList[:] = self.learningRateList[0]/10.0

		self.learningRateList[8:] = self.learningRateList[0]/5.0
		self.learningRateList[12:] = self.learningRateList[0]/10.0
		self.learningRateList[16:] = self.learningRateList[0]/20.0
		self.psnrRecord = []
		# build model -- define placeholder
		self.X = tf.placeholder(tf.float32, [self.batchSize, 512, 512, 1], name = 'low_dose_image')
		self.Y = tf.placeholder(tf.float32, [self.batchSize, 512, 512, 1], name = 'original_image')
		self.isTraining = tf.placeholder(tf.bool, name = 'is_training')
		self.learningRate = tf.placeholder(tf.float32, name = 'learning_rate')
		# build model -- build NN
		self.Gx, self.attMap, self.E0x, self.E1x, self.E2x, self.E3x, self.E4x, self.Wx, self.Wy = generatorNN(NNinput = self.X, nnOption = G_option, name = 'G_block', isTraining = self.isTraining, reuse = False, device = self.avaliableGPUlist[:], verbose = paraDict.get('verbose', True))
		self.Dy = discriminatorNN(NNinput = self.Y, nnOption = D_option, name = 'D_block', isTraining = self.isTraining, reuse = False, device = self.avaliableGPUlist[0])
		self.DGx = discriminatorNN(NNinput = self.Gx, nnOption = D_option, name = 'D_block', isTraining = self.isTraining, reuse = True, device = self.avaliableGPUlist[0])
		# build model -- create loss
		self.G_loss = lossFunc(X = self.X, DGx = self.DGx, Gx = self.Gx, Y = self.Y, Dy = self.Dy, batchSize = self.batchSize, lossOption = lossOption + '_G')
		self.D_loss = lossFunc(X = self.X, DGx = self.DGx, Gx = self.Gx, Y = self.Y, Dy = self.Dy, batchSize = self.batchSize, lossOption = lossOption + '_D')
		self.DG_loss = lossFunc(X = self.X, DGx = self.DGx, Gx = self.Gx, Y = self.Y, Dy = self.Dy, batchSize = self.batchSize, lossOption = lossOption + '_DG')
		self.psnr = tf_psnr(self.Gx, self.Y)
		self.DTest_DGx = self.DGx
		self.DTest_Dy = self.Dy

		t_vars = tf.trainable_variables()
		d_vars = [var for var in t_vars if 'D_' in var.name]
		g_vars = [var for var in t_vars if 'G_' in var.name]

		with tf.device(self.avaliableGPUlist[0]):
			self.D_solver = tf.train.AdamOptimizer(self.learningRate).minimize(self.D_loss, var_list = d_vars)
			self.G_solver = tf.train.AdamOptimizer(self.learningRate).minimize(self.G_loss + 20*self.DG_loss, var_list = g_vars)

		self.sess.run(tf.global_variables_initializer())
		print('Info: Successfully init Testmodel(NNmodel)')

	def train(self, trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean = None, evaluationImgLoader_noisy = None, **paraDict):
		self.stdTrain(trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean, evaluationImgLoader_noisy, \
						epochNumPH = 1, batchSizePH = self.batchSize, learningRatePH = self.learningRate, learningRateListPH = self.learningRateList, \
						cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining, \
						sessRunParaDictList = [	{'name':['D_solver', 'D_loss', 'DGx', 'Dy'], 'para':[self.D_solver, self.D_loss, self.DTest_DGx, self.DTest_Dy]}, \
												{'name':['D_solver', 'D_loss', 'DGx', 'Dy'], 'para':[self.D_solver, self.D_loss, self.DTest_DGx, self.DTest_Dy]}, \
												{'name':['G_solver', 'G_loss'], 'para':[self.G_solver, self.G_loss]} ], \
						**paraDict)
		self.stdTrain(trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean, evaluationImgLoader_noisy, \
						epochNumPH = 1, batchSizePH = self.batchSize, learningRatePH = self.learningRate, learningRateListPH = self.learningRateList, \
						cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining, \
						sessRunParaDictList = [	{'name':['D_solver', 'D_loss', 'DGx', 'Dy'], 'para':[self.D_solver, self.D_loss, self.DTest_DGx, self.DTest_Dy]} ], \
						**paraDict)
		self.stdTrain(trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean, evaluationImgLoader_noisy, \
						epochNumPH = self.epochNum, batchSizePH = self.batchSize, learningRatePH = self.learningRate, learningRateListPH = self.learningRateList, \
						cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining, \
						sessRunParaDictList = [	{'name':['D_solver', 'D_loss', 'DGx', 'Dy'], 'para':[self.D_solver, self.D_loss, self.DTest_DGx, self.DTest_Dy]}, \
												{'name':['D_solver', 'D_loss', 'DGx', 'Dy'], 'para':[self.D_solver, self.D_loss, self.DTest_DGx, self.DTest_Dy]}, \
												{'name':['G_solver', 'G_loss'], 'para':[self.G_solver, self.G_loss]} ], \
						**paraDict)

	def evaluate(self, evaluationImgLoader_clean, evaluationImgLoader_noisy, **paraDict):
		self.stdEvaluate(evaluationImgLoader_clean = evaluationImgLoader_clean[:], evaluationImgLoader_noisy = evaluationImgLoader_noisy[:], cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining)
	
	def generate(self, img, **paraDict):
		return self.stdGenerate(img = img, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining)

	# def testNN(self, img, **paraDict):
	# 	if self.verbose: print('Info: NNmodel.NN716.testNN() start.')
	# 	# debug.autoImgXCheck(imgX = imgBatch, info = 'in GANmodel.generate(), imgBatch before NN, should be float')
	# 	imgBatch = imgFormatConvert.reshapeImgToImgBatch(img).astype(np.float32)
	# 	YImgBatch, XImgBatch, GxImgBatch, attMap, E0xImgBatch, E1xImgBatch, E2xImgBatch, E3xImgBatch, E4xImgBatch, Wx, Wy = self.sess.run([self.Y, self.X, self.Gx, self.attMap, self.E0x, self.E1x, self.E2x, self.E3x, self.E4x, self.Wx, self.Wy], feed_dict = {self.X: imgBatch, self.isTraining: False})
	# 	# GxImgBatch, E1xImgBatch, E2xImgBatch, E3xImgBatch = self.sess.run([self.Gx, self.E1x, self.E2x, self.E3x], feed_dict = {self.X: imgBatch, self.isTraining: False})
	# 	# debug.autoImgXCheck(imgX = outputImgBatchList[0], info = 'in GANmodel.generate(), outputImgBatch after NN, should be float')

	# 	print('Info: shape of attMap is:', attMap.shape)
	# 	# (1, 24649, 24649)

	# 	imgPainter.autoPaint(GxImgBatch, path = './temp/Gx.jpg', reportImageInfo = True, fixDataOverflow = True)
	# 	imgPainter.autoPaint(XImgBatch, path = './temp/X.jpg', reportImageInfo = True, fixDataOverflow = True)
	# 	imgPainter.autoPaintPlus(E0xImgBatch, path = './temp/E0x_0.jpg', channel = 0, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E0xImgBatch, path = './temp/E0x_1.jpg', channel = 1, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E0xImgBatch, path = './temp/E0x_2.jpg', channel = 2, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E0xImgBatch, path = './temp/E0x_3.jpg', channel = 3, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E1xImgBatch, path = './temp/E1x_0.jpg', channel = 0, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E1xImgBatch, path = './temp/E1x_1.jpg', channel = 1, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E1xImgBatch, path = './temp/E1x_2.jpg', channel = 2, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E1xImgBatch, path = './temp/E1x_3.jpg', channel = 3, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E2xImgBatch, path = './temp/E2x_0.jpg', channel = 0, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E2xImgBatch, path = './temp/E2x_1.jpg', channel = 1, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E2xImgBatch, path = './temp/E2x_2.jpg', channel = 2, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E2xImgBatch, path = './temp/E2x_3.jpg', channel = 3, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E3xImgBatch, path = './temp/E3x_0.jpg', channel = 0, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E3xImgBatch, path = './temp/E3x_1.jpg', channel = 1, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E3xImgBatch, path = './temp/E3x_2.jpg', channel = 2, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E3xImgBatch, path = './temp/E3x_3.jpg', channel = 3, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E4xImgBatch, path = './temp/E4x_0.jpg', channel = 0, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E4xImgBatch, path = './temp/E4x_1.jpg', channel = 1, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E4xImgBatch, path = './temp/E4x_2.jpg', channel = 2, reportImageInfo = True, fixDataOverflow = False)
	# 	imgPainter.autoPaintPlus(E4xImgBatch, path = './temp/E4x_3.jpg', channel = 3, reportImageInfo = True, fixDataOverflow = False)

	# 	print('Result: Weight of short cut is:', Wx)
	# 	print('Result: Weight of attention block is:', Wy)

	# 	pixBlockID = int(157*(157*1/2+2) + 157*1/2+2)
	# 	attVec = attMap[0][pixBlockID]
	# 	attMat = np.reshape(a = attVec, newshape = (157, 157, 1))
	# 	imgPainter.autoPaintPlus(attMat, path = './temp/attForSomePix-whitePix.jpg', channel = 0, reportImageInfo = True, fixDataOverflow = False)

	# 	pixBlockID = int(157*(157*1/10) + 157*1/10)
	# 	attVec = attMap[0][pixBlockID]
	# 	attMat = np.reshape(a = attVec, newshape = (157, 157, 1))
	# 	imgPainter.autoPaintPlus(attMat, path = './temp/attForSomePix-blackPix.jpg', channel = 0, reportImageInfo = True, fixDataOverflow = False)

	# 	# np.savetxt('attention_map.txt', attMap[0])

	# 	return






'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''


class NN721NSA(NNmodel):

	def __init__(self, sess, G_option = '721NSA_G', D_option = '721NSA_D', lossOption = 'NN721_loss', **paraDict):
		super().__init__(sess = sess, modelName = 'NNmodel721', checkpointFolderPath = paraDict.get('checkpointFolderPath', './checkpoint'), resultFolderPath = paraDict.get('resultFolderPath', './result'), verbose = paraDict.get('verbose', True))

		self.usePatch = paraDict.get('usePatch', False)
		self.globalStep = 0
		self.batchSize = paraDict.get('batchSize', 1.0)
		self.epochNum = 50
		self.learningRateList = 0.0001 * np.ones(self.epochNum)
		self.learningRateList[8:] = self.learningRateList[0]/5.0
		self.learningRateList[12:] = self.learningRateList[0]/10.0
		self.learningRateList[16:] = self.learningRateList[0]/20.0
		self.psnrRecord = []
		# build model -- define placeholder
		self.X = tf.placeholder(tf.float32, [None, None, None, 1], name = 'low_dose_image')
		self.Y = tf.placeholder(tf.float32, [None, None, None, 1], name = 'original_image')
		# self.X = tf.placeholder(tf.float32, [self.batchSize * self.patchSize, None, None, 1], name = 'low_dose_image')
		# self.Y = tf.placeholder(tf.float32, [self.batchSize * self.patchSize, None, None, 1], name = 'original_image')
		# self.X = tf.placeholder(tf.float32, [self.batchSize, 512, 512, 1], name = 'low_dose_image')
		# self.Y = tf.placeholder(tf.float32, [self.batchSize, 512, 512, 1], name = 'original_image')
		self.isTraining = tf.placeholder(tf.bool, name = 'is_training')
		self.learningRate = tf.placeholder(tf.float32, name = 'learning_rate')
		# build model -- build NN
		self.Gx, self.attMap, self.E0x, self.E1x, self.E2x, self.E3x, self.E4x, self.Wx, self.Wy = generatorNN(NNinput = self.X, nnOption = G_option, name = 'G_block', isTraining = self.isTraining, reuse = False, device = self.avaliableGPUlist[:])
		self.Dy = discriminatorNN(NNinput = self.Y, nnOption = D_option, name = 'D_block', isTraining = self.isTraining, reuse = False, device = self.avaliableGPUlist[0])
		self.DGx = discriminatorNN(NNinput = self.Gx, nnOption = D_option, name = 'D_block', isTraining = self.isTraining, reuse = True, device = self.avaliableGPUlist[0])
		# build model -- create loss
		self.G_loss = lossFunc(X = self.X, DGx = self.DGx, Gx = self.Gx, Y = self.Y, Dy = self.Dy, batchSize = self.batchSize, lossOption = lossOption + '_G')
		self.D_loss = lossFunc(X = self.X, DGx = self.DGx, Gx = self.Gx, Y = self.Y, Dy = self.Dy, batchSize = self.batchSize, lossOption = lossOption + '_D')
		self.DG_loss = lossFunc(X = self.X, DGx = self.DGx, Gx = self.Gx, Y = self.Y, Dy = self.Dy, batchSize = self.batchSize, lossOption = lossOption + '_DG')
		self.psnr = tf_psnr(self.Gx, self.Y)
		self.DTest_DGx = self.DGx
		self.DTest_Dy = self.Dy

		t_vars = tf.trainable_variables()
		d_vars = [var for var in t_vars if 'D_' in var.name]
		g_vars = [var for var in t_vars if 'G_' in var.name]

		with tf.device(self.avaliableGPUlist[0]):
			self.D_solver = tf.train.AdamOptimizer(self.learningRate).minimize(self.D_loss, var_list = d_vars)
			self.G_solver = tf.train.AdamOptimizer(self.learningRate).minimize(self.G_loss + 20*self.DG_loss, var_list = g_vars)

		self.sess.run(tf.global_variables_initializer())
		print('Info: Successfully init Testmodel(NNmodel)')

	def train(self, trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean = None, evaluationImgLoader_noisy = None, **paraDict):
		self.stdTrain(trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean, evaluationImgLoader_noisy, \
						epochNumPH = 1, batchSizePH = self.batchSize, learningRatePH = self.learningRate, learningRateListPH = self.learningRateList, \
						cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining, \
						sessRunParaDictList = [	{'name':['D_solver', 'D_loss', 'DGx', 'Dy'], 'para':[self.D_solver, self.D_loss, self.DTest_DGx, self.DTest_Dy]}, \
												{'name':['D_solver', 'D_loss', 'DGx', 'Dy'], 'para':[self.D_solver, self.D_loss, self.DTest_DGx, self.DTest_Dy]}, \
												{'name':['G_solver', 'G_loss'], 'para':[self.G_solver, self.G_loss]} ], \
						usePatch = self.usePatch, \
						**paraDict)
		self.stdTrain(trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean, evaluationImgLoader_noisy, \
						epochNumPH = 1, batchSizePH = self.batchSize, learningRatePH = self.learningRate, learningRateListPH = self.learningRateList, \
						cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining, \
						sessRunParaDictList = [	{'name':['D_solver', 'D_loss', 'DGx', 'Dy'], 'para':[self.D_solver, self.D_loss, self.DTest_DGx, self.DTest_Dy]} ], \
						usePatch = self.usePatch, \
						**paraDict)
		self.stdTrain(trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean, evaluationImgLoader_noisy, \
						epochNumPH = self.epochNum, batchSizePH = self.batchSize, learningRatePH = self.learningRate, learningRateListPH = self.learningRateList, \
						cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining, \
						sessRunParaDictList = [	{'name':['D_solver', 'D_loss', 'DGx', 'Dy'], 'para':[self.D_solver, self.D_loss, self.DTest_DGx, self.DTest_Dy]}, \
												{'name':['D_solver', 'D_loss', 'DGx', 'Dy'], 'para':[self.D_solver, self.D_loss, self.DTest_DGx, self.DTest_Dy]}, \
												{'name':['G_solver', 'G_loss'], 'para':[self.G_solver, self.G_loss]} ], \
						usePatch = self.usePatch, \
						**paraDict)

	def evaluate(self, evaluationImgLoader_clean, evaluationImgLoader_noisy, **paraDict):
		self.stdEvaluate(evaluationImgLoader_clean = evaluationImgLoader_clean[:], evaluationImgLoader_noisy = evaluationImgLoader_noisy[:], cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining)
	
	def generate(self, img, **paraDict):
		return self.stdGenerate(img = img, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining)




'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''


# class NN814(NNmodel):
# 	# adapted SRResGAN

# 	def __init__(self, sess, G_option = '814_G', D_option = '814_D', lossOption = 'NN721_loss', **paraDict):
# 		super().__init__(sess = sess, modelName = 'NNmodel814', checkpointFolderPath = paraDict.get('checkpointFolderPath', './checkpoint'), resultFolderPath = paraDict.get('resultFolderPath', './result'), verbose = paraDict.get('verbose', True))


# 	def train(self, trainImgLoader_clean, trainImgLoader_noisy, evaluationImgLoader_clean = None, evaluationImgLoader_noisy = None, **paraDict):
# 		debug.imgLoaderCheck(imgLoader = trainImgLoader_clean, info = 'trainImgLoader_clean')
# 		debug.imgLoaderCheck(imgLoader = evaluationImgLoader_clean, info = 'evaluationImgLoader_clean')

		

# 	def evaluate(self, evaluationImgLoader_clean, evaluationImgLoader_noisy, **paraDict):
# 		self.stdEvaluate(evaluationImgLoader_clean = evaluationImgLoader_clean[:], evaluationImgLoader_noisy = evaluationImgLoader_noisy[:], cleanImgPH = self.Y, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining)
	
# 	def generate(self, img, **paraDict):
# 		return self.stdGenerate(img = img, noisyImgPH = self.X, generatedImgPH = self.Gx, isTrainingPH = self.isTraining)

# 	def testNN(self, img, **paraDict):
# 		if self.verbose: print('Info: NNmodel.NN716.testNN() start.')
		




'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''
'''=========================================================================================================================================='''


