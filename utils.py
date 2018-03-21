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

fig = plt.figure()

LIMIT = 5

def showMaps(P, W, Map, poseHistoryX, poseHistoryY):
	# Plot map
	m = ((Map['map'] - LIMIT)/(LIMIT * 1.5))
	plt.imshow(np.exp(m), cmap="hot")

	# Plot robot path
	plt.plot(poseHistoryY, poseHistoryX, 'bo', markersize=1)

	fig.canvas.draw()
	fig.canvas.flush_events()
	fig.show()

def mapCorrelationSimple(Map, poseXcell, poseYcell, z_occCells):
	cell_x_start = 0#int(max(poseXcell - 100, 0))
	cell_x_end = Map['sizex'] - 1#int(min(poseXcell + 100, Map['sizex'] - 1))

	cell_y_start = 0#int(max(poseYcell - 100, 0))
	cell_y_end = Map['sizey'] - 1#int(min(poseYcell + 100, Map['sizey'] - 1))

	# roi = Map['map'][cell_x_start : cell_x_end, cell_y_start : cell_y_end]

	zocc_x = np.logical_and(z_occCells[0] <= cell_x_end, z_occCells[0] >= cell_x_start)
	zocc_y = np.logical_and(z_occCells[1] <= cell_y_end, z_occCells[1] >= cell_y_start)
	zocci = np.logical_and(zocc_x, zocc_y)
	return Map['map'][z_occCells[0, zocci], z_occCells[1, zocci]]

	# mi = (Map['map'] - LIMIT)/LIMIT > -1
	# copyMap = np.copy(Map['map'])
	# copyMap[mi] = 1
	# copyMap[~mi] = 0
	# return copyMap[z_occCells[0, zocci], z_occCells[1, zocci]]

def calCorrelationValue(c):
	return np.sum(c)

def smartPlus(a, b):
	a = a.reshape(3, 1)
	b = b.reshape(3, 1)
	
	res = np.zeros((3, 1))	
	thetaT = a[2, 0]
	res[2, 0] = b[2, 0] + a[2, 0]
	R = np.array([[np.cos(thetaT), -np.sin(thetaT)], [np.sin(thetaT), np.cos(thetaT)]]).reshape(2,2)
	res[0:2] = a[0:2].reshape(2, 1) + np.matmul(R, b[0:2]).reshape(2, 1)
	return res

def smartMinus(a, b):
	a = a.reshape(3, 1)
	b = b.reshape(3, 1)

	res = np.zeros((3, 1))	
	thetaT = b[2, 0]
	res[2, 0] = (a[2, 0] - b[2, 0])
	R = np.array([[np.cos(thetaT), np.sin(thetaT)], [-np.sin(thetaT), np.cos(thetaT)]]).reshape(2,2)
	res[0:2] = np.matmul(R, a[0:2] - b[0:2])
	return res

def getGaussianNoise():
	mean = [0, 0, 0]
	# return np.array(mean)
	cov = np.array([[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.005]])
	return np.random.multivariate_normal(mean, cov)

def getCellsFromPhysicalGlobalCoordinates(z_globalFrame, Map):
	# z_globalFrame is assumed to be (3 X z_globalFrame.shape[1])
	# convert from meters to cells
	x_cell = np.ceil((z_globalFrame[0] - Map['xmin']) / Map['res']).astype(np.int16) - 1
	y_cell = np.ceil((z_globalFrame[1] - Map['ymin']) / Map['res']).astype(np.int16) - 1
	return np.concatenate([x_cell.reshape(1, x_cell.size), y_cell.reshape(1, y_cell.size)], axis=0)

def transformLidarH2B(transfAngles, z, angles):
	yaw_head, pitch_head, roll_body, pitch_body, yaw_body = transfAngles
	# p1 is cartesian coordinate in neck frame 
	# (to go from head to neck frame, just add neck distance in z-direction)
	# p1 is (3 X z.size)
	z_size = z.size
	p1 = np.zeros((3, z_size))
	p1[0] = np.multiply(z, np.cos(angles))
	p1[1] = np.multiply(z, np.sin(angles))
	p1[2] = 0.15 # Lidar to neck distance, to transform to neck frame for head frame

	rotz = np.array([[np.cos(yaw_head), -np.sin(yaw_head), 0], \
		[np.sin(yaw_head), np.cos(yaw_head), 0], \
		[0, 0, 1]])

	roty = np.array([[np.cos(pitch_head), 0, np.sin(pitch_head)], \
		[0, 1, 0], \
		[-np.sin(pitch_head), 0, np.cos(pitch_head)]])

	# p2 is in body frame
	# p2 is (3 X z_size)
	p2 = np.matmul(np.matmul(rotz, roty), p1.T.reshape(z_size, 3, 1)).reshape(z_size, 3).T
	p2[2] = p2[2] + 1.26 # Neck to body frame (distance of 1.26m neck to body origin at ground)

	# p2 is (3 X z_size)
	return p2

def useBodyRollPitch(transfAngles, pose, z_bodyFrame):
	yaw_head, pitch_head, roll_body, pitch_body, yaw_body = transfAngles
	R_bP = np.array([[np.cos(pitch_body), 0, np.sin(pitch_body)], \
		[0, 1, 0], \
		[-np.sin(pitch_body), 0, np.cos(pitch_body)]])
	R_bR = np.array([[1, 0, 0], \
		[0, np.cos(yaw_body), -np.sin(yaw_body)], \
		[0, np.sin(yaw_body), np.cos(yaw_body)]])

	z_bodyFrame[2, :] = z_bodyFrame[2, :] - (0.93 - 0.16)

	z_size = z_bodyFrame.shape[1]
	z_bodyFrame = np.matmul(np.matmul(R_bP, R_bR), z_bodyFrame.T.reshape(z_size, 3, 1)).reshape(z_size, 3).T
	z_bodyFrame[2, :] = z_bodyFrame[2, :] + (0.93 - 0.16)
	return z_bodyFrame

def transformLidarB2G(transfAngles, pose, z_bodyFrame):
	pose = pose.reshape(3)
	x = pose[0]
	y = pose[1]
	theta = pose[2]

	# z_bodyFrame = useBodyRollPitch(transfAngles, pose, z_bodyFrame)

	tB2G = np.array([[np.cos(theta), -np.sin(theta), 0, x], \
		[np.sin(theta), np.cos(theta), 0, y], \
		[0, 0, 1, 0], \
		[0, 0, 0, 1]])

	# z_bodyFrame is assumed to be (3 X z_bodyFrame.shape[1]), changing to (4 X z_bodyFrame.shape[1])
	z_bodyFrame = np.concatenate((z_bodyFrame, np.ones((1, z_bodyFrame.shape[1]))), axis=0)
	z_bfSize = z_bodyFrame.shape[1]
	z_globalFrame = np.matmul(tB2G, z_bodyFrame.T.reshape(z_bfSize, 4, 1)).reshape(z_bfSize, 4).T
	return z_globalFrame[0:3]

def transformLidarH2G(transfAngles, pose, z, angles):
	# Transform z to global frame from head frame
	z_bodyFrame = transformLidarH2B(transfAngles, z, angles)
	z_globalFrame = transformLidarB2G(transfAngles, pose, z_bodyFrame)
	return z_globalFrame

def getGroundIndices(transfAngles, z, angles):
	z_bodyFrame = transformLidarH2B(transfAngles, z, angles)
	return (z_bodyFrame[2] < (0 + 0.1)) # Taking some ground threshold

def getGoodIndicesLaserScan(transfAngles, z, angles):
	# Indices (of angles or z) that are too close, far, or hit the ground
	indValid = np.logical_and((z < 30), (z > 0.1))
	indGround = getGroundIndices(transfAngles, z, angles)
	return np.logical_and(indValid, ~indGround)

def updateMapLogOdds(m, z_occCells, z_freeCells):
	# z_occCells is (2 X z_occCells.shape[1]) array
	# z_freeCells is (2 X z_freeCells.shape[1]) array
	z_occCells = z_occCells.astype(np.int16)
	z_freeCells = z_freeCells.astype(np.int16)

	deltaLogOddsFreeCell = -0.008
	xis = np.logical_and((z_freeCells[0] >= 0), (z_freeCells[0] < m['sizex']))
	yis = np.logical_and((z_freeCells[1] >= 0), (z_freeCells[1] < m['sizey']))
	valIndices = np.logical_and(xis, yis)
	m['map'][z_freeCells[0][valIndices], z_freeCells[1][valIndices]] = \
	m['map'][z_freeCells[0][valIndices], z_freeCells[1][valIndices]] + deltaLogOddsFreeCell

	deltaLogOddsOccCell = 0.05
	xis = np.logical_and((z_occCells[0] >= 0), (z_occCells[0] < m['sizex']))
	yis = np.logical_and((z_occCells[1] >= 0), (z_occCells[1] < m['sizey']))
	valIndices = np.logical_and(xis, yis)
	m['map'][z_occCells[0][valIndices], z_occCells[1][valIndices]] = \
	m['map'][z_occCells[0][valIndices], z_occCells[1][valIndices]] + deltaLogOddsOccCell

	lowerBound = -LIMIT
	upperBound = LIMIT
	ind = m['map'] > upperBound
	m['map'][ind] = upperBound
	ind = m['map'] < lowerBound
	m['map'][ind] = lowerBound

	return m

def getMLEParticle(P, W):
	return P[np.argmax(W)]

def getDataAtSameTimestamp(lidar, joint):
	jointTimeSeries = joint['ts'][0]

	lidarTimesSeries = np.zeros(len(lidar))
	for i in range(len(lidar)):
		lidarTimesSeries[i] = lidar[i]['t'][0][0]

	ji, li = getSynchArrays(jointTimeSeries.reshape(1, jointTimeSeries.size), lidarTimesSeries.reshape(1, lidarTimesSeries.size))

	joint['ts'] = joint['ts'][:, ji]
	joint['rpy'] = joint['rpy'][:, ji]
	joint['head_angles'] = joint['head_angles'][:, ji]
	joint['acc'] = joint['acc'][:, ji]
	joint['gyro'] = joint['gyro'][:, ji]

	lidar = [lidar[i] for i in li]

	return lidar, joint