

from tensorflow.python.client import device_lib

def patch_GPUdistribution_for_NNstructure_discriminatorNN_simpleGAN():
	local_device_protos = device_lib.list_local_devices()
	avaliableGPUlist = [ x.name.replace('device:', '').lower() for x in local_device_protos if 'GPU' in x.name ]
	# print('Info: In patches.patch_GPUdistribution_for_NNstructure_discriminatorNN_simpleGAN()')
	# print('      avaliable GPU: %s' % str(avaliableGPUlist))
	useTrainingDistribution = True if len(avaliableGPUlist) >= 4 else False
	return avaliableGPUlist[1] if useTrainingDistribution else avaliableGPUlist[0]
