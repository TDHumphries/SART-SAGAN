
import argparse
import astra
import numpy as np

import dataLoader
import debug
import imgFormatConvert
import imgPainter
import parameters


def createSinogram_batch(imgLoader, sinogramFolderPath, option = 'default_option', **paraDict):

	parameters.sinogramCreationParameterInitialization(option = option)
	
	numpix = parameters.sino_numpix #512
	dx = parameters.sino_dx #1.0
	numbin = parameters.sino_numbin  #729
	numtheta = parameters.sino_numtheta #900
	theta_range = parameters.sino_theta_range #[0, 180]
	geom = parameters.sino_geom #'fanflat'
	counts = parameters.sino_counts #1e4
	dso = parameters.sino_dso #100.0
	dod = parameters.sino_dod #100.0
	fan_angle = parameters.sino_fan_angle #35.0
	whetherAddNoise = parameters.sino_whetherAddNoise #True

	sinogramFilePathList = []


	vol_geom = astra.create_vol_geom(numpix, numpix)

	theta_range = np.deg2rad(theta_range)
	angles = theta_range[0] + np.linspace(0, numtheta - 1, numtheta, False) * (theta_range[1] - theta_range[0]) / numtheta

	if geom == 'parallel':
		print('Info: geom == parallel, create projection geometry')
		proj_geom = astra.create_proj_geom(geom, 1.0, numbin, angles)
	elif geom == 'fanflat':
		print('Info: geom == fanflat, create projection geometry')
		# convert to mm for astra
		dso *= 10
		dod *= 10
		# compute tan of 1/2 the fan angle
		ft = np.tan(np.deg2rad(fan_angle / 2))
		# width of one detector pixel, calculated based on fan angle
		det_width = 2 * (dso + dod) * ft / numbin
		proj_geom = astra.create_proj_geom(geom, det_width, numbin, angles, dso, dod)

	p = astra.create_projector('cuda', proj_geom, vol_geom)

	for img, pureName in imgLoader:
		img = imgFormatConvert.reshapeImgToMatrix(img = img)

		# debug.autoImgXCheck(imgX = img, info = 'in sinogram.createSinogram_batch(), should be float img')

		print('Info: Create sinogram for image %s' % str(pureName))
		sino_id, sino = astra.create_sino(img, p)
		# 255 level img is required here

		sino = sino.astype('float32')
		sino = sino / 255
		# debug.autoImgXCheck(imgX = sino, info = 'sino')
		# imgPainter.autoPaint(imgX = sino, path = '../sino-mid.png')


		# normalize to pixel size
		sino = sino * dx
		# exponentiate
		sino = counts * np.exp(-sino)
		# sino = counts * np.exp(sino)
		# add noise
		sino = np.random.poisson(sino) if whetherAddNoise is True else sino
		sino = imgPainter.fixMatrixDataOverflow(matrix = sino)
		# return to log domain
		sino = -np.log(sino/counts)
		sino = np.float32(sino)

		sino = imgPainter.fixMatrixDataOverflow(matrix = sino)


		sino.tofile(sinogramFolderPath + '/' + pureName + '_sino(' + parameters.sino_option + ').flt')
		imgPainter.autoPaint(imgX = sino, path = sinogramFolderPath + '/' + pureName + '_sino(' + parameters.sino_option + ').png')
		sinogramFilePathList.append(sinogramFolderPath + '/' + pureName + '_sino(' + parameters.sino_option + ').flt')

		# print('sinogramFilePathList == %s' % str(sinogramFilePathList))

	sinogramImgLoader = dataLoader.FltLoader(fltFilePathList = sinogramFilePathList)

	return sinogramFilePathList, sinogramImgLoader

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--func', dest = 'func', type = str, default = 'default_func')
	parser.add_argument('--inFile', dest = 'inFile', type = str, default = 'default_in')
	parser.add_argument('--outFile', dest = 'outFile', type = str, default = 'default_out')
	parser.add_argument('--dataType', dest = 'dataType', type = str, default = 'png')
	parser.add_argument('--option', dest = 'option', type = str, default = 'default_option')
	args = parser.parse_args()

	print('args.inFile == %s' % str(args.inFile))

	parameters.globalParametersInitialization()

	imgLoader = dataLoader.PngLoader(pngFilePathList = [args.inFile]) if args.dataType == 'png' else dataLoader.FltLoader(fltFilePathList = [args.inFile]) if args.dataType == 'flt' else None
	sinogramFilePathList, sinogramImgLoader = createSinogram_batch(imgLoader = imgLoader, sinogramFolderPath = '../temp', option = args.option)

	debug.imgLoaderCheck(imgLoader = sinogramImgLoader, info = 'sinogramImgLoader')
	sinoImg = sinogramImgLoader.loadImg()
	imgPainter.autoPaint(imgX = sinoImg, path = args.outFile)

