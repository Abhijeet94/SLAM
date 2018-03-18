import sys
sys.path.insert(0,'Proj4_2018_Train')
sys.path.insert(0,'Proj4_2018_Train/MapUtils')
from scipy import io
import cv2, os
import math
import numpy as np
import pdb
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import MapUtils as MU
import MapUtils2 as MU2
import load_data as ld
import p4_util as util
################################################################################################

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

def replayLidarData():
	lidar0 = ld.get_lidar('Proj4_2018_Train/data/train_lidar0')
	pdb.set_trace()
	util.replay_lidar(lidar0)

def randomExperiment():
	joint = ld.get_joint('Proj4_2018_Train/data/train_joint3')
	key_names_joint = ['acc', 'ts', 'rpy', 'gyro', 'pos', 'ft_l', 'ft_r', 'head_angles']
	print joint['head_angles'].shape
	# print util.get_joint_index(joint)

################################################################################################

def showMaps(P, W, Map):
	pass

def calCorrelationValue(c):
	return np.sum(c)

def smartPlus(a, b):
	a = a.reshape(3, 1)
	b = b.reshape(3, 1)
	
	res = np.zeros((3, 1))	
	thetaT = b[2]
	res[2, 0] = b[2] + a[2]
	R = np.array([[np.cos(thetaT), -np.sin(thetaT)], [np.sin(thetaT), np.cos(thetaT)]])
	res[0:2] = b[0:2].reshape(2, 1) + np.matmul(R, a[0:2]).reshape(2, 1)
	return res

def smartMinus(a, b):
	a = a.reshape(3, 1)
	b = b.reshape(3, 1)

	res = np.zeros((3, 1))	
	thetaT = b[2]
	res[2, 0] = a[2] - b[2]
	R = np.array([[np.cos(thetaT), np.sin(thetaT)], [-np.sin(thetaT), np.cos(thetaT)]])
	res[0:2] = np.matmul(R, a[0:2] - b[0:2])
	return res

def getGaussianNoise():
	mean = [0, 0, 0]
	cov = np.array([[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.4]])
	return np.random.multivariate_normal(mean, cov)

def getCellsFromPhysicalGlobalCoordinates(z_globalFrame, Map):
	# z_globalFrame is assumed to be (3 X z_globalFrame.shape[1])
	# convert from meters to cells
	x_cell = np.ceil((z_globalFrame[0] - Map['xmin']) / Map['res']).astype(np.int16) - 1
	y_cell = np.ceil((z_globalFrame[1] - Map['ymin']) / Map['res']).astype(np.int16) - 1
	return np.concatenate((x_cell, y_cell), axis=0)

def transformLidarH2B(transfAngles, z, angles):
	yaw_head, pitch_head, roll_body, pitch_body, yaw_body = transfAngles

	# p1 is cartesian coordinate in neck frame 
	# (to go from head to neck frame, just add neck distance in z-direction)
	# p1 is (3 X z.size)
	p1 = np.zeros((3, z.size))
	p1[0] = np.multiply(z, np.cos(angles))
	p1[1] = np.multiply(z, np.sin(angles))
	p1[2] = -0.15 # Lidar to neck distance, to transform to neck frame for head frame

	rotz = np.array([[np.cos(yaw_head), -np.sin(yaw_head), 0], \
		[np.sin(yaw_head), np.cos(yaw_head), 0], \
		[0, 0, 1]])

	roty = np.array([[np.cos(pitch_head), 0, np.sin(pitch_head)], \
		[0, 1, 0], \
		[-np.sin(pitch_head), 0, np.cos(pitch_head)]])

	# p2 is in body frame
	# p2 is (3 X z.size)
	p2 = np.matmul(np.matmul(rotz, roty), p1.T).T
	p2[2] = p2[2] - 1.26 # Neck to body frame (distance of 1.26m neck to body origin at ground)

	# p2 is (3 X z.size)
	return p2

def transformLidarB2G(pose, z_bodyFrame):
	pose = pose.reshape(3)
	x = pose[0]
	y = pose[1]
	theta = pose[2]

	tB2G = np.array([[np.cos(theta), -np.sin(theta), 0, x], \
		[np.sin(theta), np.cos(theta), 0, y], \
		[0, 0, 1, 0], \
		[0, 0, 0, 1]])

	# z_bodyFrame is assumed to be (3 X z_bodyFrame.shape[1])
	z_bodyFrame = np.concatenate((z_bodyFrame, np.ones((1, z_bodyFrame.shape[1]))), axis=0)
	z_globalFrame = np.matmul(tB2G, z_bodyFrame.T).T
	return z_globalFrame[0:3]

def transformLidarH2G(transfAngles, pose, z, angles):
	# Transform z to global frame from head frame
	z_bodyFrame = transformLidarH2B(transfAngles, z, angles)
	z_globalFrame = transformLidarB2G()
	return z_globalFrame

def getGroundIndices(transfAngles, pose, z, angles):
	z_bodyFrame = transformLidarH2B(transfAngles, z, angles)
	return (z_bodyFrame[2] < (0 + 0.1)) # Taking some ground threshold

def getGoodIndicesLaserScan(transfAngles, pose, z, angles):
	# Indices (of angles or z) that are too close, far, or hit the ground
	indValid = np.logical_and((z < 30), (z > 0.1))
	indGround = getGroundIndices(transfAngles, pose, z, angles)
	return np.logical_and(indValid, ~indGround)

def updateMapLogOdds(m, z_occCells, z_freeCells):
	pass

def getMLEParticle(P, W):
	return P[np.argmax(W)]

def getDataAtSameTimestamp(lidar, joint):
	jointTimeSeries = joint['ts'][0]

	lidarTimesSeries = np.zeros(len(lidar))
	for i in range(len(lidar)):
		lidarTimesSeries[i] = lidar[i]['t'][0][0]

	ji, li = getSynchArrays(jointTimeSeries, lidarTimesSeries)

	joint['ts'][:] = joint['ts'][:][ji]
	joint['rpy'][:] = joint['rpy'][:][ji]
	joint['head_angles'][:] = joint['head_angles'][:][ji]
	joint['acc'][:] = joint['acc'][:][ji]
	joint['gyro'][:] = joint['gyro'][:][ji]

	lidar = lidar[li]

	return lidar, joint