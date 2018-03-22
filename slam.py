import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import matplotlib.image as mpimg
import time
from scipy.misc import logsumexp

sys.path.insert(0,'Proj4_2018_Train')
sys.path.insert(0,'Proj4_2018_Train/MapUtils')

import load_data as ld
import p4_util as util
import MapUtils as MU
import MapUtils2 as MU2
from utils import *

################################################################################################

def initSlam(lidar, joint, numParticles):
	P = np.zeros((numParticles, 3))
	W = np.ones(numParticles) * (1.0/numParticles)

	# init MAP
	MAP = {}
	MAP['res']   = 0.05 #meters
	MAP['xmin']  = -22  #meters
	MAP['ymin']  = -22
	MAP['xmax']  =  22
	MAP['ymax']  =  22 
	MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
	MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
	MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']))

	return P, W, MAP

################################################################################################

# - z - represents the laser scan (head frame) at time 't'
# - p - is the MLE particle at time 't'
# transfAngles - (head angles, rpy) used to transform the laser scan from head to global frame
# - m - The map at time 't'
# Returns: m_(t+1) - The map at time 't+1'
def mapping(z, p, transfAngles, m):
	angles = np.array(np.arange(135,-135.25,-0.25)*np.pi/180.)

	# Remove scan points that are too close, far, or hit the ground
	zi = getGoodIndicesLaserScan(transfAngles, z, angles)
	z = z[zi]
	angles = angles[zi]

	# Transform z to global frame
	z_globalFrame = transformLidarH2G(transfAngles, p, z, angles)
	z_occCells = getCellsFromPhysicalGlobalCoordinates(z_globalFrame, m)

	# Use getMapCellsFromRay() to get free cells
	poseXcell = np.ceil((p[0] - m['xmin']) / m['res']) - 1
	poseYcell = np.ceil((p[1] - m['ymin']) / m['res']) - 1
	z_freeCells = MU2.getMapCellsFromRay(poseXcell, poseYcell, z_occCells[0].tolist(), z_occCells[1].tolist())

	# Update log-odds ratio in m
	m_new = updateMapLogOdds(m, z_occCells, z_freeCells)
	
	return m_new

# - P - Pose of all 'n' particles at time 't'
# - o_t - odometery at time 't' - confirm whether relative and which reference
# - o_t1 - odometery at time 't+1' - confirm whether relative and which reference
# Returns: Pose of all 'n' particles at time 't+1'
def localizationPrediction(P, o_t, o_t1):
	PP = np.zeros(P.shape)

	# For each particle
	for i in range(P.shape[0]):
		# Get some random noise for each particle
		odometeryChange = smartMinus(o_t1, o_t)
		noise = getGaussianNoise().reshape(3, 1) #* np.linalg.norm(odometeryChange)

		# p_t+1 = p_t ++ (o_t+1 -- o_t) ++ noise
		PP[i] = smartPlus(P[i], smartPlus(odometeryChange, noise)).reshape(3)
		# print 'With noise: ' + str(PP[i])
		# print 'Without noise: ' + str(smartPlus(P[i], smartMinus(o_t1, o_t)).reshape(3))
		# print '---------'

	return PP

# - P - Pose of all 'n' particles at time 't+1'
# - W - Weight of all 'n' particles at time 't'
# - z - represents the laser scan (relative to head frame) at time 't'
# - m - The map at time 't+1'
# transfAngles - (head angles, rpy) used to transform the laser scan from head to global frame
# Returns: w_(t+1) - Weight of all 'n' particles at time 't+1'
def localizationUpdate(P, W, z, m, transfAngles):
	logWW = np.log(W)
	angles = np.array(np.arange(135,-135.25,-0.25)*np.pi/180.)

	# Remove scan points that are too close, far, or hit the ground
	zi = getGoodIndicesLaserScan(transfAngles, z, angles)
	z = z[zi]
	angles = angles[zi]

	# For each particle
	for i in range(P.shape[0]):
		
		# Transform z to global frame
		z_globalFrame = transformLidarH2G(transfAngles, P[i], z, angles)

		# Compute correlation of the scan points with the map
		# x_im = np.arange(m['xmin'], m['xmax'] + m['res'], m['res'])
		# y_im = np.arange(m['ymin'], m['ymax'] + m['res'], m['res'])
		# x_range = np.arange(P[i][0] - 0.05, P[i][0] + 0.05, 0.025) 
		# y_range = np.arange(P[i][1] - 0.05, P[i][1] + 0.05, 0.025) 
		# c = MU2.mapCorrelation(m['map'], x_im, y_im, z_globalFrame, x_range, y_range)[0]
		#####
		poseXcell = np.ceil((P[i][0] - m['xmin']) / m['res']) - 1
		poseYcell = np.ceil((P[i][1] - m['ymin']) / m['res']) - 1
		z_occCells = getCellsFromPhysicalGlobalCoordinates(z_globalFrame, m)
		c = mapCorrelationSimple(m, poseXcell, poseYcell, z_occCells)
		#####

		# Update the weight of the particle
		logWW[i] = logWW[i] + calCorrelationValue(c)
		# print 'Correlation value for particle ' + str(i) + ' is:\t' + str(calCorrelationValue(c))

	logWW = logWW - logsumexp(logWW)
	return np.exp(logWW)

# - P - Pose of all 'n' particles at time 't+1'
# - W - Weight of all 'n' particles at time 't+1'
# - numParticles - number of particles to return
# Returns: (P, W) - new particles and their weights
def resampleIfNeeded(P, W, numParticles):
	nEffective = np.square(np.sum(W)) * 1.0 / (np.sum(np.square(W)))
	nThreshold = numParticles * 0.4

	if nEffective < nThreshold:
		# print 'n_eff is low: ' + str(nEffective)
		PP = np.zeros(P.shape)
		WW = np.ones(numParticles) * (1.0/numParticles)

		for i in range(numParticles):
			PP[i] = P[np.argmin((np.cumsum(W) * 1.0 / np.sum(W)) < np.random.uniform())]

		P = PP
		W = WW
	# else:
	# 	print 'oioioioio: ' + str(nEffective)

	return (P, W)

def slam(lidar, joint):
	numParticles = 100
	lidar, joint = getDataAtSameTimestamp(lidar, joint)
	P, W, Map = initSlam(lidar, joint, numParticles)
	poseHistoryX = []
	poseHistoryY = []
	step = 5
	startPos =  0#40 * 500

	for t in xrange(startPos, len(lidar) - step * 2, step):
		# print 'Time step: ' + str(t) + ', Pose:\t' + str(getMLEParticle(P, W))
		z_headFrame = lidar[t]['scan'][0]
		trans = (joint['head_angles'][0][t], joint['head_angles'][1][t], joint['rpy'][0][t], joint['rpy'][1][t], joint['rpy'][2][t])
		# print 'Yaw: ' + str(smartMinus(lidar[t+step]['pose'][0], lidar[t]['pose'][0])[2])
		# print 'Body - ' + str(joint['rpy'][0][t]) + ', ' + str(joint['rpy'][1][t])

		p_mle = getMLEParticle(P, W) 
		poseHistoryX.append(np.ceil((p_mle[0] - Map['xmin']) / Map['res']) - 1)
		poseHistoryY.append(np.ceil((p_mle[1] - Map['xmin']) / Map['res']) - 1)

		start = time.time()
		Map = mapping(z_headFrame, p_mle, trans, Map)
		# print 'Mapping time: ' + str(time.time() - start)

		start = time.time()
		P = localizationPrediction(P, lidar[t]['pose'][0], lidar[t+step]['pose'][0])
		# print 'Location Prediction time: ' + str(time.time() - start)

		# print 'localization update started'
		start = time.time()
		W = localizationUpdate(P, W, z_headFrame, Map, trans)
		# print 'Location Update time: ' + str(time.time() - start)
		# print 'localization update done'

		if t % 500 == 0:
			start = time.time()
			showMaps(P, W, Map, poseHistoryX, poseHistoryY)
			print 'Map drawing time: ' + str(time.time() - start)

		P, W = resampleIfNeeded(P, W, numParticles)

	return P, W, Map, poseHistoryX, poseHistoryY

################################################################################################

def runSlam():
	lidarFile = 'Proj4_2018_Train/data/train_lidar1'
	jointFile = 'Proj4_2018_Train/data/train_joint1'

	plt.ion()
	# plt.show()

	lidar = ld.get_lidar(lidarFile)
	joint = ld.get_joint(jointFile)

	P, W, Map, poseHistoryX, poseHistoryY = slam(lidar, joint)
	showMaps(P, W, Map, poseHistoryX, poseHistoryY)
	print 'Done'
	plt.show(block=True)

if __name__ == "__main__":
	# replayLidarData()
	# randomExperiment()

	runSlam()	