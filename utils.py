from scipy import io
import cv2, os
import math
import numpy as np
import transforms3d
import pdb
from time import sleep

def getSynchArrays(gt, pred):
	gtSynch = []
	predSync = []

	gtIndex = 0
	predIndex = 0

	while True:
		if gtIndex + 1 >= gt.size or predIndex + 1 >= pred.size:
			break

		if gt[0, gtIndex] < pred[0, predIndex]:
			if gt[0, gtIndex + 1] < pred[0, predIndex]:
				gtIndex = gtIndex + 1
			else:
				if abs(gt[0, gtIndex] - pred[0, predIndex]) < abs(gt[0, gtIndex + 1] - pred[0, predIndex]):
					gtSynch.append(gtIndex)
					predSync.append(predIndex)
					gtIndex = gtIndex + 1
					predIndex = predIndex + 1
				else:
					gtSynch.append(gtIndex + 1)
					predSync.append(predIndex)
					gtIndex = gtIndex + 2
					predIndex = predIndex + 1
		else:
			if pred[0, predIndex + 1] < gt[0, gtIndex]:
				predIndex = predIndex + 1
			else:
				if abs(gt[0, gtIndex] - pred[0, predIndex]) < abs(gt[0, gtIndex] - pred[0, predIndex+1]):
					gtSynch.append(gtIndex)
					predSync.append(predIndex)
					gtIndex = gtIndex + 1
					predIndex = predIndex + 1
				else:
					gtSynch.append(gtIndex + 1)
					predSync.append(predIndex)
					gtIndex = gtIndex + 1
					predIndex = predIndex + 2

	return gtSynch, predSync