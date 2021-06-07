
import argparse
import astra
import copy
import glob
import numpy as np
import os
import tensorflow as tf

import dataLoader
import debug
import imgFormatConvert
import imgPainter
import NNmodel
import parameters




def create_projector(geom, numbin, angles, dso, dod, fan_angle, vol_geom):
	if geom == 'parallel':
		proj_geom = astra.create_proj_geom(geom, 1.0, numbin, angles)
	elif geom == 'fanflat':
		dso *=10; dod *=10;                         #convert to mm for astra
		ft = np.tan( np.deg2rad(fan_angle / 2) )    #compute tan of 1/2 the fan angle
		det_width = 2 * (dso + dod) * ft / numbin  #width of one detector pixel, calculated based on fan angle

		proj_geom = astra.create_proj_geom(geom, det_width, numbin, angles, dso, dod)

	p = astra.create_projector('cuda',proj_geom,vol_geom)
	# p = astra.create_projector(proj_type = 'cuda', proj_geom = self.projectorGeometry, vol_geom = self.volumeGeometry)

	return p

def sart_batch(sinogramImgLoader, outputFolderPath, NNfunction, option = 'default_option', iterNum = None, paintSteps = True, ns = None):

	parameters.SARTreconstructionParameterInitialization(option = option)
	parameters.recon_numits = parameters.recon_numits if iterNum is None else iterNum
	parameters.recon_paintSteps = paintSteps

	reconImgFilePathList = []
	sinogramImgLoader = copy.deepcopy(sinogramImgLoader)

	for img, imgPureName in sinogramImgLoader:
		sino = imgFormatConvert.reshapeImgToMatrix(img = img)
		reconImgFilePath, _ = sart(sino = sino, sinoPureName = imgPureName + ('_NNSART' if NNfunction is not None else '_oriSART'), outputFolderPath = outputFolderPath, NNfunction = NNfunction, ns = ns)
		reconImgFilePathList.append(reconImgFilePath)

	reconImgLoader = dataLoader.FltLoader(fltFilePathList = reconImgFilePathList) if parameters.recon_paintFlt is True else dataLoader.PngLoader(pngFilePathList = reconImgFilePathList)

	return reconImgFilePathList, reconImgLoader




def MNNsart_batch(sinogramImgLoader, outputFolderPath, option = 'default_option', mnnOrder = 'sart10|(NN721)./checkpoint|return', paintSteps = True, imgMiddleName = '_MNNSART', returnMatF = False, verbose = False, ns = None):
	
	mnnOrderList = mnnOrder.split('|')

	parameters.SARTreconstructionParameterInitialization(option = option)
	parameters.recon_paintSteps = paintSteps
	sinogramImgLoader = copy.deepcopy(sinogramImgLoader)
	reconImgFilePathList = []
	matFList_list = []

	if ns is not None:
		parameters.recon_ns = ns

	for img, imgPureName in sinogramImgLoader:
		print('Info: reconstruction.MNNsart_batch(): Handle img %s' % str(imgPureName))
		mnnBlockNum = 0
		parameters.recon_paintFlt = False
		parameters.recon_paintPng = False
		matFList = []

		sino = imgFormatConvert.reshapeImgToMatrix(img = img)
		# print('Debug: reconstruction.MNNsart_batch() line ~74: sino ==', sino)
		# print('sino.shape =', sino.shape)
		# mipa
		_, matF = sart(sino = sino, sinoPureName = imgPureName + imgMiddleName + '_block%d' % mnnBlockNum, outputFolderPath = outputFolderPath, NNfunction = None, x0mat = None, numits = 1)
		mnnBlockNum += 1

		for mnnStepOrder in mnnOrderList:
			print('Info: MNNsart: mnnStepOrder == %s' % mnnStepOrder)
			
			if mnnStepOrder.find('sart') == 0:
				try:
					sartStepIterNum = int(mnnStepOrder.split('sart')[1].split('ns')[0])
					sartStepNsNum = int(mnnStepOrder.split('ns')[1])
				except IndexError:
					print('WARNING! Unknow sart step setting in mnnStepOrder: %s' % str(mnnStepOrder))
					print('         we\'ll reset sartStepIterNum = 20, sartStepNsNum = 10')
					sartStepIterNum = 20
					sartStepNsNum = 10
				_, matF = sart(sino = sino, sinoPureName = imgPureName + imgMiddleName + '_block%d' % mnnBlockNum, outputFolderPath = outputFolderPath, NNfunction = None, x0mat = matF, numits = sartStepIterNum, ns = sartStepNsNum)
				mnnBlockNum += 1
			elif mnnStepOrder.find('(gan)') == 0:
				# imgPainter.autoPaint(imgX = matF, path = './temp/' + imgPureName + '_block' + str(mnnBlockNum) + '_beforeGAN.png')
				nnStepCkptPath = str(mnnStepOrder.split('(gan)')[1])
				sessConfig = tf.ConfigProto()
				sessConfig.gpu_options.allow_growth = True
				sessConfig.gpu_options.per_process_gpu_memory_fraction = 1
				with tf.Session(config = sessConfig) as sess:
					projectModel = NNmodel.GANmodel(sess = sess, checkpointFolderPath = nnStepCkptPath, G_option = 'simpleGAN_G', D_option = 'simpleGAN_D', lossOption = 'l2_loss')
					checkpointStatus = projectModel.loadModel()
					if checkpointStatus is False:
						print('ERROR! In reconstruction.MNNsart_batch()')
						print('       Unable to load checkpoint. checkpointFolderPath == %s' % str(nnStepCkptPath))
						print('       FATAL ERROR, FORCE EXIT')
						exit()
					NNfunction = lambda x: imgFormatConvert.reshapeImgToMatrix(projectModel.generate(img = imgFormatConvert.reshapeMatrixToImg(imgPainter.fixMatrixDataOverflow(x))))
					matF = NNfunction(matF)
					# debug.autoImgXCheck(imgX = matF)
				tf.reset_default_graph()
				# imgPainter.autoPaint(imgX = matF, path = './temp/' + imgPureName + '_block' + str(mnnBlockNum) + '_afterGAN.png')
			elif mnnStepOrder.find('(testGAN)') == 0:
				# imgPainter.autoPaint(imgX = matF, path = './temp/' + imgPureName + '_block' + str(mnnBlockNum) + '_beforeGAN.png')
				nnStepCkptPath = str(mnnStepOrder.split('(testGAN)')[1])
				sessConfig = tf.ConfigProto()
				sessConfig.gpu_options.allow_growth = True
				sessConfig.gpu_options.per_process_gpu_memory_fraction = 1
				with tf.Session(config = sessConfig) as sess:
					projectModel = NNmodel.Testmodel(sess = sess, checkpointFolderPath = nnStepCkptPath)
					checkpointStatus = projectModel.loadModel()
					if checkpointStatus is False:
						print('ERROR! In reconstruction.MNNsart_batch()')
						print('       Unable to load checkpoint. checkpointFolderPath == %s' % str(nnStepCkptPath))
						print('       FATAL ERROR, FORCE EXIT')
						exit()
					NNfunction = lambda x: imgFormatConvert.reshapeImgToMatrix(projectModel.generate(img = imgFormatConvert.reshapeMatrixToImg(imgPainter.fixMatrixDataOverflow(x))))
					matF = NNfunction(matF)
					# debug.autoImgXCheck(imgX = matF)
				tf.reset_default_graph()
			elif mnnStepOrder.find('(NN721)') == 0:
				# imgPainter.autoPaint(imgX = matF, path = './temp/' + imgPureName + '_block' + str(mnnBlockNum) + '_beforeGAN.png')
				nnStepCkptPath = str(mnnStepOrder.split('(NN721)')[1])
				sessConfig = tf.ConfigProto()
				sessConfig.gpu_options.allow_growth = True
				sessConfig.gpu_options.per_process_gpu_memory_fraction = 1
				with tf.Session(config = sessConfig) as sess:
					projectModel = NNmodel.NN721(sess = sess, checkpointFolderPath = nnStepCkptPath, verbose = False)
					checkpointStatus = projectModel.loadModel()
					if checkpointStatus is False:
						print('ERROR! In reconstruction.MNNsart_batch()')
						print('       Unable to load checkpoint. checkpointFolderPath == %s' % str(nnStepCkptPath))
						print('       FATAL ERROR, FORCE EXIT')
						exit()
					NNfunction = lambda x: imgFormatConvert.reshapeImgToMatrix(projectModel.generate(img = imgFormatConvert.reshapeMatrixToImg(imgPainter.fixMatrixDataOverflow(x))))
					matF = NNfunction(matF)
					# debug.autoImgXCheck(imgX = matF)
				tf.reset_default_graph()
			elif mnnStepOrder.find('(NN721NSA)') == 0:
				# imgPainter.autoPaint(imgX = matF, path = './temp/' + imgPureName + '_block' + str(mnnBlockNum) + '_beforeGAN.png')
				nnStepCkptPath = str(mnnStepOrder.split('(NN721NSA)')[1])
				sessConfig = tf.ConfigProto()
				sessConfig.gpu_options.allow_growth = True
				sessConfig.gpu_options.per_process_gpu_memory_fraction = 1
				with tf.Session(config = sessConfig) as sess:
					projectModel = NNmodel.NN721NSA(sess = sess, checkpointFolderPath = nnStepCkptPath)
					checkpointStatus = projectModel.loadModel()
					if checkpointStatus is False:
						print('ERROR! In reconstruction.MNNsart_batch()')
						print('       Unable to load checkpoint. checkpointFolderPath == %s' % str(nnStepCkptPath))
						print('       FATAL ERROR, FORCE EXIT')
						exit()
					NNfunction = lambda x: imgFormatConvert.reshapeImgToMatrix(projectModel.generate(img = imgFormatConvert.reshapeMatrixToImg(imgPainter.fixMatrixDataOverflow(x))))
					matF = NNfunction(matF)
					# debug.autoImgXCheck(imgX = matF)
				tf.reset_default_graph()
			# elif mnnStepOrder.find('(NN1007cycle)') == 0:
			# 	# imgPainter.autoPaint(imgX = matF, path = './temp/' + imgPureName + '_block' + str(mnnBlockNum) + '_beforeGAN.png')
			# 	nnStepCkptPath = str(mnnStepOrder.split('(NN1007cycle)')[1])
			# 	sessConfig = tf.ConfigProto()
			# 	sessConfig.gpu_options.allow_growth = True
			# 	sessConfig.gpu_options.per_process_gpu_memory_fraction = 1
			# 	with tf.Session(config = sessConfig) as sess:
			# 		projectModel = NNmodel.NN1007cycle(sess = sess, checkpointFolderPath = nnStepCkptPath, repeatGPUlist = True)
			# 		checkpointStatus = projectModel.loadModel()
			# 		if checkpointStatus is False:
			# 			print('ERROR! In reconstruction.MNNsart_batch()')
			# 			print('       Unable to load checkpoint. checkpointFolderPath == %s' % str(nnStepCkptPath))
			# 			print('       FATAL ERROR, FORCE EXIT')
			# 			exit()
			# 		NNfunction = lambda x: imgFormatConvert.reshapeImgToMatrix(projectModel.generate(img = imgFormatConvert.reshapeMatrixToImg(imgPainter.fixMatrixDataOverflow(x))))
			# 		matF = NNfunction(matF)
			# 		# debug.autoImgXCheck(imgX = matF)
			# 	tf.reset_default_graph()
			# elif mnnStepOrder.find('(NN1007cycleP)') == 0:
			# 	# imgPainter.autoPaint(imgX = matF, path = './temp/' + imgPureName + '_block' + str(mnnBlockNum) + '_beforeGAN.png')
			# 	nnStepCkptPath = str(mnnStepOrder.split('(NN1007cycleP)')[1])
			# 	sessConfig = tf.ConfigProto()
			# 	sessConfig.gpu_options.allow_growth = True
			# 	sessConfig.gpu_options.per_process_gpu_memory_fraction = 1
			# 	with tf.Session(config = sessConfig) as sess:
			# 		projectModel = NNmodel.NN1007cycleP(sess = sess, checkpointFolderPath = nnStepCkptPath, repeatGPUlist = True)
			# 		checkpointStatus = projectModel.loadModel()
			# 		if checkpointStatus is False:
			# 			print('ERROR! In reconstruction.MNNsart_batch()')
			# 			print('       Unable to load checkpoint. checkpointFolderPath == %s' % str(nnStepCkptPath))
			# 			print('       FATAL ERROR, FORCE EXIT')
			# 			exit()
			# 		NNfunction = lambda x: imgFormatConvert.reshapeImgToMatrix(projectModel.generate(img = imgFormatConvert.reshapeMatrixToImg(imgPainter.fixMatrixDataOverflow(x))))
			# 		matF = NNfunction(matF)
			# 		# debug.autoImgXCheck(imgX = matF)
			# 	tf.reset_default_graph()
			# elif mnnStepOrder.find('(NN1107)') == 0:
			# 	# imgPainter.autoPaint(imgX = matF, path = './temp/' + imgPureName + '_block' + str(mnnBlockNum) + '_beforeGAN.png')
			# 	nnStepCkptPath = str(mnnStepOrder.split('(NN1107)')[1])
			# 	sessConfig = tf.ConfigProto()
			# 	sessConfig.gpu_options.allow_growth = True
			# 	sessConfig.gpu_options.per_process_gpu_memory_fraction = 1
			# 	with tf.Session(config = sessConfig) as sess:
			# 		projectModel = NNmodel.NN1107(sess = sess, checkpointFolderPath = nnStepCkptPath)
			# 		checkpointStatus = projectModel.loadModel()
			# 		if checkpointStatus is False:
			# 			print('ERROR! In reconstruction.MNNsart_batch()')
			# 			print('       Unable to load checkpoint. checkpointFolderPath == %s' % str(nnStepCkptPath))
			# 			print('       FATAL ERROR, FORCE EXIT')
			# 			exit()
			# 		NNfunction = lambda x: imgFormatConvert.reshapeImgToMatrix(projectModel.generate(img = imgFormatConvert.reshapeMatrixToImg(imgPainter.fixMatrixDataOverflow(x))))
			# 		matF = NNfunction(matF)
			# 		# debug.autoImgXCheck(imgX = matF)
			# 	tf.reset_default_graph()
			elif mnnStepOrder.find('TVsart') == 0:
				try:
					sartStepIterNum = int(mnnStepOrder.split('TVsart')[1].split('ns')[0])
					sartStepNsNum = int(mnnStepOrder.split('ns')[1])
				except IndexError:
					print('WARNING! Unknow TVsart step setting in mnnStepOrder: %s' % str(mnnStepOrder))
					print('         we\'ll reset sartStepIterNum = 20, sartStepNsNum = 10')
					sartStepIterNum = 20
					sartStepNsNum = 10
				_, matF = sart(sino = sino, sinoPureName = imgPureName + imgMiddleName + '_block%d' % mnnBlockNum, outputFolderPath = outputFolderPath, NNfunction = None, x0mat = matF, numits = sartStepIterNum, ns = sartStepNsNum, use_sup = True)
				mnnBlockNum += 1
			elif mnnStepOrder.find('return') == 0:
				parameters.recon_paintFlt = True
				parameters.recon_paintPng = True
				reconImgFilePath, _ = sart(sino = sino, sinoPureName = imgPureName + imgMiddleName, outputFolderPath = outputFolderPath, NNfunction = lambda x: x, x0mat = matF, numits = 1, ns = 1)
				reconImgFilePathList.append(reconImgFilePath)
				# return reconImgFilePath, dataLoader.FltLoader(fltFilePathList = [reconImgFilePath])
			else:
				print('ERROR! Unable to solve mnnStepOrder')
				print('       mnnStepOrder == %s' % str(mnnStepOrder))
				print('       FATAL ERROR, FORCE EXIT')
				exit()

			if returnMatF is True:
				matFList.append(matF)

		if returnMatF is True:
			matFList_list.append(matFList)

	if returnMatF is True:
		print('Info: In reconstruction.MNNsart_batch(): matFList_list will be returned after reconImgLoader.')
		return reconImgFilePathList, dataLoader.FltLoader(fltFilePathList = reconImgFilePathList), matFList_list

	return reconImgFilePathList, dataLoader.FltLoader(fltFilePathList = reconImgFilePathList)

def pureSART_batch(sinogramImgLoader, outputFolderPath, noiseOption, iterNum, ns):
	parameters.SARTreconstructionParameterInitialization(option = noiseOption)
	parameters.recon_paintSteps = False
	parameters.recon_paintFlt = False
	parameters.recon_paintPng = False
	parameters.recon_ns = ns
	parameters.recon_numits = iterNum

	reconImgFilePathList = []

	for sinoImg, sinoImgPureName in sinogramImgLoader:
		print('Info: pure SART recon -', sinoImgPureName)
		parameters.recon_paintFlt = False
		parameters.recon_paintPng = False
		sino = imgFormatConvert.reshapeImgToMatrix(img = sinoImg)
		_, matF = sart(sino = sino, sinoPureName = sinoImgPureName, outputFolderPath = outputFolderPath, NNfunction = None, x0mat = None, numits = iterNum-1, ns = ns)
		parameters.recon_paintFlt = True
		parameters.recon_paintPng = True
		reconImgFilePath, _ = sart(sino = sino, sinoPureName = sinoImgPureName, outputFolderPath = outputFolderPath, NNfunction = None, x0mat = matF, numits = 1, ns = ns)
		reconImgFilePathList.append(reconImgFilePath)

	return dataLoader.FltLoader(fltFilePathList = reconImgFilePathList)

def grad_TV(img, numpix):
	#pdb.set_trace()
	epsilon = 1e-6
	ind_m1 = np.arange(numpix)
	ind_m2 = [(i + 1) % numpix for i in ind_m1]
	ind_m0 = [(i - 1) % numpix for i in ind_m1]

	m2m1 = np.ix_(ind_m2,ind_m1)
	m1m2 = np.ix_(ind_m1,ind_m2)
	m0m1 = np.ix_(ind_m0,ind_m1)
	m1m0 = np.ix_(ind_m1,ind_m0)

	diff1 = ( img[m2m1] - img) ** 2
	diff2 = ( img[m1m2] - img) ** 2
	diffttl = np.sqrt( diff1 + diff2 + epsilon**2)
	TV = np.sum(diffttl)

	dTV = -1/diffttl * (img[m2m1]-2*img + img[m1m2]) + \
			1/diffttl[m0m1] * (img-img[m0m1]) + \
			1/diffttl[m1m0] * (img-img[m1m0])
	return TV, dTV	

def sart(sino, sinoPureName, outputFolderPath, NNfunction, x0mat = None, numits = None, ns = None, **paraDict):

	numOfIterationApplyNN = 10
	x0file = parameters.recon_x0file #''
	xtruefile = parameters.recon_xtruefile #''
	numpix = parameters.recon_numpix #512
	dx = parameters.recon_dx #1.0
	numbin = parameters.recon_numbin #729
	numtheta = parameters.recon_numtheta #900
	ns = parameters.recon_ns if ns is None else ns #10
	numits = parameters.recon_numits if numits is None else numits #200
	beta = parameters.recon_beta #1.0
	theta_range = parameters.recon_theta_range #[0.0, 180.0]
	geom = parameters.recon_geom #'fanflat'
	dso = parameters.recon_dso #100.0
	dod = parameters.recon_dod #100.0
	fan_angle = parameters.recon_fan_angle #35.0

	eps = np.finfo(float).eps

	use_sup = paraDict.get('use_sup', False)
	if use_sup is True:
		gamma = 0.9995
		N = np.uint8(20)
		alpha = 0.5

	sino = sino * 255
	# unfinished: check sino, sino should be 255 level image
	# debug.autoImgXCheck(imgX = sino, info = 'in reconstruction.sart(), should be 255 level img')


	# create projection geometry
	vol_geom = astra.create_vol_geom(numpix, numpix)

	#generate array of angular positions
	theta_range = np.deg2rad(theta_range) #convert to radians
	angles = theta_range[0] + np.linspace(0,numtheta-1,numtheta,False)*(theta_range[1]-theta_range[0])/numtheta #

	# if x0file == '':
	# 	f = np.zeros((numpix,numpix))
	# else:
	# 	f = np.fromfile(x0file,dtype='f')
	# 	f = f.reshape(numpix,numpix)
	f = x0mat if x0mat is not None else np.zeros((numpix,numpix))

	if xtruefile == '':
		calc_error = False
		xtrue = np.zeros((numpix,numpix))
	else:
		xtrue = np.fromfile(xtruefile,dtype='f')
		xtrue = xtrue.reshape(numpix,numpix)
		calc_error = True

	#create projectors and normalization terms (corresponding to diagonal matrices M and D) for each subset of projection data
	P, Dinv, D_id, Minv, M_id = [None]*ns,[None]*ns,[None]*ns,[None]*ns,[None]*ns
	for j in range(ns):
		ind1 = range(j,numtheta,ns);
		p = create_projector(geom,numbin,angles[ind1],dso,dod,fan_angle, vol_geom)


		D_id[j], Dinv[j] = astra.create_backprojection(np.ones((numtheta//ns,numbin)),p)
		M_id[j], Minv[j] = astra.create_sino(np.ones((numpix,numpix)),p)
		#avoid division by zero, also scale M to pixel size
		Dinv[j] = np.maximum(Dinv[j],eps)
		Minv[j] = np.maximum(Minv[j]*dx,eps)
		P[j] = p

	for k in range(numits):

		if use_sup is True:
			#pdb.set_trace()
			g,dg = grad_TV(f,numpix)
			g_old = g
			dg = -dg / (np.linalg.norm(dg,'fro') + eps)
			for j in range(N):
				while True:
					f_tmp = f + alpha * dg
					g_new = grad_TV(f_tmp,numpix)[0]
					alpha = alpha*gamma
					if g_new <= g_old:
						f = f_tmp
						break

				dg = grad_TV(f,numpix)[1]
				dg = -dg / (np.linalg.norm(dg,'fro') + eps)

		for j in range(ns):
			ind1 = range(j,numtheta,ns);
			p = P[j]
			fp_id, fp = astra.create_sino(f,p)      #forward projection step
			# fp_id, fp = astra.create_sino(data = f, proj_id = p, returnData = True, gpuIndex = None)
			diffs = (sino[ind1,:] - fp*dx)/Minv[j]                  #should do elementwise division
			bp_id, bp = astra.create_backprojection(diffs,p)
			ind2 = np.abs(bp) > 1e3
			bp[ind2] = 0             #get rid of spurious large values
			f = f + beta * bp/Dinv[j];                   #update f
			astra.data2d.delete(fp_id)
			astra.data2d.delete(bp_id)

		f = np.maximum(f,eps);                  #set any negative values to small positive value

		# Add denoiser here!
		whetherUseNN = (k > numits - 3) or (k == numits - 1)
		if parameters.recon_paintSteps is True: # and whetherUseNN is True:
			imgPainter.autoPaint(imgX = f * 255, path = outputFolderPath + '/steps/' + sinoPureName + '_recon_%s.png' % str(k), reportImageInfo = False)
		f = NNfunction(f) if (NNfunction is not None) and (whetherUseNN is True) else f



		#compute residual and error (if xtrue is provided)
		fp = np.zeros((numtheta, numbin))
		for j in range(ns):
			ind = range(j,numtheta,ns)
			p = P[j]
			fp_tempid, fp_temp = astra.create_sino(f,p)
			fp[ind,:] = fp_temp*dx
			astra.data2d.delete(fp_tempid)

		res = np.linalg.norm(fp-sino,'fro') / np.linalg.norm(sino,'fro')
		#pdb.set_trace()
		if calc_error:
			err = np.linalg.norm(f-xtrue,'fro')/np.linalg.norm(xtrue,'fro')
			print('Iteration #{0:d}: Residual = {1:1.4f}\tError = {2:1.4f}'.format(k+1,res,err))
		else:
			# print('Iteration #{0:d}: Residual = {1:1.4f}'.format(k+1,res))
			pass

	#save image
	if parameters.recon_paintFlt is True:
		f = np.float32(f)
		f.tofile(outputFolderPath + '/' + sinoPureName + '_recon.flt')

	if parameters.recon_paintPng is True:
		imgPainter.autoPaint(imgX = f * 255, path = outputFolderPath + '/' + sinoPureName + '_recon.png', reportImageInfo = True, fixDataOverflow = True)
	# unfinish: check f here
	# debug.autoImgXCheck(imgX = f, info = 'in tail of sart(), f should be float value img')


	for j in range(ns):
		astra.data2d.delete(D_id[j])
		astra.data2d.delete(M_id[j])
		astra.projector.delete(P[j])

	return (outputFolderPath + '/' + sinoPureName + ('_recon.flt' if parameters.recon_paintFlt is True else '_recon.png' if parameters.recon_paintPng is True else '_recon'), f.astype(float))


# python reconstruction.py --func singleSART --inFile a_NDCT_folder_path --outFile output_folder_path --option sparse_view_60 --x0matPath a_png_CTimg_file_or_neglect_this_parameter


if __name__ == '__main__':
	

	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--func', dest = 'func', type = str, default = 'default_func')
	parser.add_argument('--inFile', dest = 'inFile', type = str, default = 'default_in')
	parser.add_argument('--outFile', dest = 'outFile', type = str, default = 'default_out')
	parser.add_argument('--dataType', dest = 'dataType', type = str, default = 'png')
	parser.add_argument('--option', dest = 'option', type = str, default = 'default_option')
	parser.add_argument('--mnnOrder', dest = 'mnnOrder', type = str, default = 'sart5|nn./checkpoint|sart20|return')
	parser.add_argument('--x0matPath', dest = 'x0matPath', type = str, default = None)
	args = parser.parse_args()

	if not os.path.exists(args.outFile):
		os.mkdir(args.outFile)
	if not os.path.exists(args.outFile + '/steps'):
		os.mkdir(args.outFile + '/steps')

	if args.func == 'multiNN':
		imgLoader = dataLoader.PngLoader(pngFilePathList = [args.inFile]) if args.dataType == 'png' else dataLoader.FltLoader(fltFilePathList = [args.inFile]) if args.dataType == 'flt' else None
		_, reconImgLoader = multiNNsart(sinogramImgLoader = imgLoader, outputFolderPath = args.outFile, option = args.option, iterNum = None, paintSteps = True)

		debug.imgLoaderCheck(imgLoader = reconImgLoader)
		imgPainter.autoPaint(imgX = reconImgLoader.loadImg(), path = '../temp/multiNNreconTest.png')

	# elif args.func == 'full':
		
	# 	import sinogram
	# 	import imgEvaluation

	# 	pngFilePathList = glob.glob(args.inFile + '/*.png')
	# 	pngFilePathList.sort()
	# 	imgLoader = dataLoader.PngLoader(pngFilePathList = [pngFilePathList[0]])
	# 	_, sinogramImgLoader = sinogram.createSinogram_batch(imgLoader = copy.deepcopy(imgLoader), sinogramFolderPath = args.outFile, option = args.option)
		
	# 	_, reconImgLoader_MNN = MNNsart_batch(sinogramImgLoader = copy.deepcopy(sinogramImgLoader), outputFolderPath = args.outFile, option = args.option, mnnOrder = args.mnnOrder, paintSteps = True)
	# 	# debug.imgLoaderCheck(imgLoader = reconImgLoader_MNN)

	# 	_, reconImgLoader_SART = sart_batch(sinogramImgLoader = sinogramImgLoader, outputFolderPath = args.outFile, NNfunction = None, option = args.option, iterNum = 200)
	# 	# debug.imgLoaderCheck(imgLoader = reconImgLoader_SART)

	# 	_, reconImgLoader_SART_NN = MNNsart_batch(sinogramImgLoader = copy.deepcopy(sinogramImgLoader), outputFolderPath = args.outFile, option = args.option, mnnOrder = 'sart199|nn../temp/ckpt_simpleGAN|return', paintSteps = True, imgMiddleName = '_SART+NN')


	# 	# debug.imgLoaderCheck(imgLoader = reconImgLoader_MNN)
	# 	psnr_MNN, _ = imgEvaluation.averagePSNR_imgLoader(imgLoader_A = copy.deepcopy(reconImgLoader_MNN), imgLoader_B = copy.deepcopy(imgLoader))
	# 	psnr_SART, _ = imgEvaluation.averagePSNR_imgLoader(imgLoader_A = copy.deepcopy(reconImgLoader_SART), imgLoader_B = copy.deepcopy(imgLoader))
	# 	psnr_SART_NN, _ = imgEvaluation.averagePSNR_imgLoader(imgLoader_A = copy.deepcopy(reconImgLoader_SART_NN), imgLoader_B = copy.deepcopy(imgLoader))

	# 	print('Result: PSNR-MNN     == %s' % str(psnr_MNN))
	# 	print('        PSNR-SART    == %s' % str(psnr_SART))
	# 	print('        PSNR-SART+NN == %s' % str(psnr_SART_NN))

	# 	imgPainter.autoPaint(imgX = reconImgLoader_MNN.loadImg(), path = '../temp/MNNreconTest.png')
	# 	imgPainter.autoPaint(imgX = reconImgLoader_SART.loadImg(), path = '../temp/noNNreconTest.png')
	# 	imgPainter.autoPaint(imgX = reconImgLoader_SART_NN.loadImg(), path = '../temp/singleNNreconTest.png')

	# elif args.func == 'mnnFull':
	# 	import sinogram
	# 	import imgEvaluation

	# 	pngFilePathList = glob.glob(args.inFile + '/*.png')
	# 	pngFilePathList.sort()
	# 	imgLoader = dataLoader.PngLoader(pngFilePathList = [pngFilePathList[0]])
	# 	_, sinogramImgLoader = sinogram.createSinogram_batch(imgLoader = copy.deepcopy(imgLoader), sinogramFolderPath = args.outFile, option = args.option)
		
	# 	# _, reconImgLoader = multiNNsart(sinogramImgLoader = copy.deepcopy(sinogramImgLoader), outputFolderPath = args.outFile, option = args.option, iterNum = None, paintSteps = True)
	# 	_, reconImgLoader = MNNsart_batch(sinogramImgLoader = sinogramImgLoader[:], outputFolderPath = args.outFile, option = args.option, mnnOrder = args.mnnOrder, paintSteps = True)
	# 	debug.imgLoaderCheck(imgLoader = reconImgLoader)

	# 	reconImgFilePathList_c, reconImgLoader_c = sart_batch(sinogramImgLoader = sinogramImgLoader, outputFolderPath = args.outFile, NNfunction = None, option = args.option, iterNum = 200)
	# 	debug.imgLoaderCheck(imgLoader = reconImgLoader_c)

	# 	debug.imgLoaderCheck(imgLoader = reconImgLoader)
	# 	psnr_NN, _ = imgEvaluation.averagePSNR_imgLoader(imgLoader_A = copy.deepcopy(reconImgLoader), imgLoader_B = copy.deepcopy(imgLoader))
	# 	psnr_c, _ = imgEvaluation.averagePSNR_imgLoader(imgLoader_A = copy.deepcopy(reconImgLoader_c), imgLoader_B = copy.deepcopy(imgLoader))

	# 	print('Result: PSNR-NN    == %s' % str(psnr_NN))
	# 	print('        PSNR-clean == %s' % str(psnr_c))

	# 	imgPainter.autoPaint(imgX = reconImgLoader.loadImg(), path = '../temp/multiNNreconTest.png')
	# 	imgPainter.autoPaint(imgX = reconImgLoader_c.loadImg(), path = '../temp/noNNreconTest.png')
	
	elif args.func == 'singleSART':
		import sinogram
		import imgEvaluation

		parameters.SARTreconstructionParameterInitialization(option = args.option)
		parameters.recon_paintSteps = False

		x0mat = None if args.x0matPath is None else imgFormatConvert.reshapeImgToMatrix(img = dataLoader.PngLoader(pngFilePathList = [args.x0matPath]).loadImg())

		pngFilePathList = glob.glob(args.inFile + '/*.png')
		pngFilePathList.sort()
		imgLoader = dataLoader.PngLoader(pngFilePathList = [pngFilePathList[0]])
		print('Info: img path ==', pngFilePathList[0])

		_, sinogramImgLoader = sinogram.createSinogram_batch(imgLoader = imgLoader[:], sinogramFolderPath = args.outFile, option = args.option)

		for img, imgPureName in sinogramImgLoader[:]:
			parameters.recon_paintPng, parameters.recon_paintFlt = True, True
			reconImgFilePath, matF = sart(sino = imgFormatConvert.reshapeImgToMatrix(img = img), sinoPureName = imgPureName + '_nakeSART', outputFolderPath = args.outFile, NNfunction = None, x0mat = x0mat, numits = 50)
			reconImgLoader = dataLoader.FltLoader(fltFilePathList = [reconImgFilePath])
			break

		_, reconImgLoader_o = MNNsart_batch(sinogramImgLoader = sinogramImgLoader[:], outputFolderPath = args.outFile, option = args.option, mnnOrder = 'sart200|return', paintSteps = False, imgMiddleName = '_oriSART')

		psnr_NN, _ = imgEvaluation.averagePSNR_imgLoader(imgLoader_A = reconImgLoader[:], imgLoader_B = imgLoader[:])
		psnr_o, _ = imgEvaluation.averagePSNR_imgLoader(imgLoader_A = reconImgLoader_o[:], imgLoader_B = imgLoader[:])

		print('Result: PSNR-partSART == %s' % str(psnr_NN))
		print('        PSNR-stdSART  == %s' % str(psnr_o))

		# imgPainter.autoPaint(imgX = reconImgLoader.loadImg(), path = '../temp/MNNreconTest.png')
		imgPainter.autoPaint(imgX = reconImgLoader_o.loadImg(), path = '../temp/oriSARTreconTest.png')

	else:

		imgLoader = dataLoader.PngLoader(pngFilePathList = [args.inFile]) if args.dataType == 'png' else dataLoader.FltLoader(fltFilePathList = [args.inFile]) if args.dataType == 'flt' else None
		reconImgFilePathList_c, reconImgLoader_c = sart_batch(sinogramImgLoader = imgLoader, outputFolderPath = args.outFile, NNfunction = None, option = args.option)

		print(reconImgFilePathList_c)
		# debug.imgLoaderCheck(imgLoader = reconImgLoader_NN, info = 'reconImgLoader_NN')
		debug.imgLoaderCheck(imgLoader = reconImgLoader_c, info = 'reconImgLoader_c')