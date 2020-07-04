import numpy as np

def DataNormalization(target, meanV=None, stdV=None):
	nData, nDim = target.shape[0], target.shape[1]
	
	output = np.zeros(shape=[nData, nDim], dtype=float)
	if meanV is None:
		meanV = np.mean(target, axis=0)
		stdV = np.std(target, axis=0, ddof=1)
		for dter in range(nData):
			output[dter,:nDim] = (target[dter,:nDim]-meanV) / stdV
	else:
		for dter in range(nData):
			output[dter,:nDim] = (target[dter,:nDim]-meanV) / stdV
	
	return output, meanV, stdV


def DataRegularization(target):
	nData, nDim = target.shape[0], target.shape[1]
	for dter in range(nData):
		temp = target[dter,:nDim]
		maxV = np.amax(temp)
		minV = np.amin(temp)
		reg_temp = 2*(temp-minV)/(maxV-minV)
		
		target[dter,:nDim] = reg_temp - np.mean(reg_temp)

	return target
