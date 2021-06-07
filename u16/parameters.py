
import numpy as np

FLT_FILE_FOLDER_PATH = '../fltData'
IMG_SIDE_LENGTH_X = 512
IMG_SIDE_LENGTH_Y = 512
IMG_CHANNEL_NUM = 1

def globalParametersInitialization(option = 'default_option'):

	# print('Debug: In parameters.globalParametersInitialization(), option == %s' % option)

	sinogramCreationParameterInitialization(option = option)
	SARTreconstructionParameterInitialization(option = option)


def sinogramCreationParameterInitialization(option = 'default_option'):

	print('Debug: In parameters.sinogramCreationParameterInitialization(), option == %s' % option)


	global sino_numpix
	global sino_dx
	global sino_numbin
	global sino_numtheta
	global sino_theta_range
	global sino_geom
	global sino_counts
	global sino_dso
	global sino_dod
	global sino_fan_angle
	global sino_whetherAddNoise
	global sino_option

	sino_numpix = 512
	sino_dx = 1.0
	sino_numbin = 729
	sino_numtheta = 900
	sino_theta_range = [0, 180]
	sino_geom = 'fanflat'
	sino_counts = 1e6
	sino_dso = 100.0
	sino_dod = 100.0
	sino_fan_angle = 35.0
	sino_whetherAddNoise = False
	sino_option = option

	if option == 'default_option':
		pass

	elif option == 'low-dose':
		sino_whetherAddNoise = True

	elif option == 'low-dose_1e4':
		sino_counts = 1e4
		sino_whetherAddNoise = True

	elif option == 'low-dose_1e5':
		sino_counts = 1e5
		sino_whetherAddNoise = True

	elif option == 'low-dose_2e5':
		sino_counts = 2e5
		sino_whetherAddNoise = True

	elif option == 'low-dose_1e6':
		sino_counts = 1e6
		sino_whetherAddNoise = True

	elif option == 'sparse_view_450':
		sino_numtheta = 450

	elif option == 'sparse_view_180':
		sino_numtheta = 180

	elif option == 'sparse_view_120':
		sino_numtheta = 120

	elif option == 'sparse_view_100':
		sino_numtheta = 100

	elif option == 'sparse_view_60':
		sino_numtheta = 60

	elif option == 'sparse_view_50':
		sino_numtheta = 50

	elif option == 'sparse_view_40':
		sino_numtheta = 40

	elif option == 'limited_angle_160':
		sino_theta_range = [0, 160]

	elif option == 'limited_angle_140':
		sino_theta_range = [0, 140]

	elif option == 'limited_angle_120':
		sino_theta_range = [0, 120]

	elif option == 'limited_angle_100':
		sino_theta_range = [0, 100]

	elif option.find('theta_range=') == 0:
		'''theta_range=0, 140'''
		startAngle = int(option[len('theta_range='):].split(',')[0])
		endAngle = int(option[len('theta_range='):].split(',')[1])
		sino_theta_range = [startAngle, endAngle]

	else:
		print('ERROR: Undefined option: %s' % str(option))
		print('       To avoid further error, force quit.')
		quit()

def SARTreconstructionParameterInitialization(option = 'default_option'):

	print('Debug: In parameters.SARTreconstructionParameterInitialization(), option == %s' % option)


	global recon_x0file
	global recon_xtruefile
	global recon_numpix
	global recon_dx
	global recon_numbin
	global recon_numtheta
	global recon_ns
	global recon_numits
	global recon_beta
	global recon_theta_range
	global recon_geom
	global recon_dso
	global recon_dod
	global recon_fan_angle
	global recon_eps
	global recon_NNlist
	global recon_paintSteps
	global recon_paintFlt
	global recon_paintPng

	recon_x0file = ''
	recon_xtruefile = ''
	recon_numpix = 512
	recon_dx = 1.0
	recon_numbin = 729
	recon_numtheta = 900
	recon_ns = 10
	recon_numits = 200
	recon_beta = 1.0
	recon_theta_range = [0.0, 180.0]
	recon_geom = 'fanflat'
	recon_dso = 100.0
	recon_dod = 100.0
	recon_fan_angle = 35.0
	recon_eps = np.finfo(float).eps
	recon_NNlist = []
	recon_paintSteps = True
	recon_paintFlt = True
	recon_paintPng = True

	if option == 'default_option':
		pass

	elif option == 'low-dose':
		pass

	elif option == 'low-dose_1e4':
		pass

	elif option == 'low-dose_1e5':
		pass

	elif option == 'low-dose_2e5':
		pass

	elif option == 'low-dose_1e6':
		pass

	elif option == 'sparse_view_450':
		recon_numtheta = 450

	elif option == 'sparse_view_180':
		recon_numtheta = 180

	elif option == 'sparse_view_120':
		recon_numtheta = 120

	elif option == 'sparse_view_100':
		recon_numtheta = 100

	elif option == 'sparse_view_60':
		recon_numtheta = 60

	elif option == 'sparse_view_50':
		recon_numtheta = 50

	elif option == 'sparse_view_40':
		recon_numtheta = 40
		
	elif option == 'limited_angle_160':
		recon_theta_range = [0.0, 160.0]

	elif option == 'limited_angle_140':
		recon_theta_range = [0.0, 140.0]

	elif option == 'limited_angle_120':
		recon_theta_range = [0.0, 120.0]

	elif option == 'limited_angle_100':
		recon_theta_range = [0.0, 100.0]

	elif option.find('theta_range=') == 0:
		'''theta_range=0, 140'''
		startAngle = int(option[len('theta_range='):].split(',')[0])
		endAngle = int(option[len('theta_range='):].split(',')[1])
		recon_theta_range = [startAngle, endAngle]

		
	else:
		print('ERROR: Undefined option: %s' % str(option))
		print('       To avoid further error, force quit.')
		quit()


class GlobalParameters(object):

	def __init__(self):
		pass