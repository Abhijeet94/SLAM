import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import matplotlib.image as mpimg

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
	MAP['xmin']  = -20  #meters
	MAP['ymin']  = -20
	MAP['xmax']  =  20
	MAP['ymax']  =  20 
	MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
	MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
	MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8)

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
	zi = getGoodIndicesLaserScan(transfAngles, p, z, angles)
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
		noise = getGaussianNoise().reshape(3, 1)

		# p_t+1 = p_t ++ (o_t+1 -- o_t) ++ noise
		PP[i] = smartPlus(smartPlus(P[i], smartMinus(o_t1, o_t)), noise).reshape(3)

	return PP

# - P - Pose of all 'n' particles at time 't+1'
# - W - Weight of all 'n' particles at time 't'
# - z - represents the laser scan (relative to head frame) at time 't'
# - m - The map at time 't+1'
# transfAngles - (head angles, rpy) used to transform the laser scan from head to global frame
# Returns: w_(t+1) - Weight of all 'n' particles at time 't+1'
def localizationUpdate(P, W, z, m, transfAngles):
	logWW = np.log(W)

	# For each particle
	for i in range(P.shape[0]):
		angles = np.array(np.arange(135,-135.25,-0.25)*np.pi/180.)

		# Remove scan points that are too close, far, or hit the ground
		zi = getGoodIndicesLaserScan(transfAngles, P[i], z, angles)
		z = z[zi]
		angles = angles[zi]

		# Transform z to global frame
		z_globalFrame = transformLidarH2G(transfAngles, P[i], z, angles)

		# Compute correlation of the scan points with the map
		x_im = np.arange(m['xmin'], m['xmax'] + m['res'], m['res'])
		y_im = np.arange(m['ymin'], m['ymax'] + m['res'], m['res'])
		x_range = np.arange(P[i][0] - 10, P[i][0] + 10, 0.05) # 20m around current pose
		y_range = np.arange(P[i][1] - 10, P[i][1] + 10, 0.05) # 20m around current pose
		c = MU2.mapCorrelation(m['map'], x_im, y_im, z_globalFrame, x_range, y_range)[0]

		# Update the weight of the particle
		logWW[i] = logWW[i] + calCorrelationValue(c)

	logWW = logWW - logsumexp(logWW)
	return np.exp(logWW)

# - P - Pose of all 'n' particles at time 't+1'
# - W - Weight of all 'n' particles at time 't+1'
# - numParticles - number of particles to return
# Returns: (P, W) - new particles and their weights
def resampleIfNeeded(P, W, numParticles):
	nEffective = 1.0/(np.sum(np.square(W)))
	nThreshold = numParticles * 0.4

	if nEffective < nThreshold:
		PP = np.zeros(P.shape)
		WW = np.ones(numParticles) * (1.0/numParticles)

		for i in range(numParticles):
			PP[i] = P[np.argmin((np.cumsum(W) * 1.0 / np.sum(W)) < np.random.random_sample())]

		P = PP
		W = WW

	return (P, W)

def slam(lidar, joint):
	numParticles = 200
	lidar, joint = getDataAtSameTimestamp(lidar, joint)
	P, W, Map = initSlam(lidar, joint, numParticles)

	for t in xrange(len(lidar) - 1):
		z_headFrame = lidar[t]['scan'][0]
		trans = (joint['head_angles'][0][t], joint['head_angles'][1][t], joint['rpy'][0][t], joint['rpy'][1][t], joint['rpy'][2][t])

		p_mle = getMLEParticle(P, W) 

		Map = mapping(z_headFrame, p_mle, trans, Map)

		P = localizationPrediction(P, lidar[t]['pose'][0], lidar[t+1]['pose'][0])

		W = localizationUpdate(P, W, z_headFrame, Map, trans)

		showMaps(P, W, Map)

		P, W = resampleIfNeeded(P, W, numParticles)

################################################################################################

def runSlam():
	lidarFile = 'Proj4_2018_Train/data/train_lidar0'
	jointFile = 'Proj4_2018_Train/data/train_joint0'

	plt.ion()
	plt.show()

	lidar = ld.get_lidar(lidarFile)
	joint = ld.get_joint(jointFile)

	slam(lidar, joint)

if __name__ == "__main__":
	# replayLidarData()
	# randomExperiment()

	runSlam()	