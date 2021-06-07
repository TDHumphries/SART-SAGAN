

import argparse
import numpy as np
import imgPainter

SIDE_LENGTH = 157


parser = argparse.ArgumentParser(description = '')
parser.add_argument('--input', dest = 'input', type = str, default = 'attentionMap.txt', help = 'attention map file')
parser.add_argument('--output', dest = 'output', type = str, default = './temp/pixAttention.jpg', help = 'attention arranged')
parser.add_argument('--pixX', dest = 'pixX', type = int, default = int(SIDE_LENGTH*2/3), help = 'X of pixel')
parser.add_argument('--pixY', dest = 'pixY', type = int, default = int(SIDE_LENGTH*2/3), help = 'Y of pixel')

args = parser.parse_args()

print('Loading array...')
attMap = np.loadtxt(args.input)
print('Load array finish')
pixBlockID = int(SIDE_LENGTH*(args.pixX) + args.pixY)
attVec = attMap[pixBlockID]
attMat = np.reshape(a = attVec, newshape = (SIDE_LENGTH, SIDE_LENGTH, 1))
imgPainter.autoPaintPlus(attMat, path = args.output, channel = 0, reportImageInfo = True, fixDataOverflow = False)


