import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import matplotlib.image as mpimg
import MapUtils as MU
import MapUtils2 as MU2

sys.path.insert(0,'Proj4_2018_Train')
sys.path.insert(0,'Proj4_2018_Train/MapUtils')

import load_data as ld
import p4_util as util
from utils import *

################################################################################################

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

def transformLidarH2G(transfAngles, p, z):
	# Transform z to global frame from head frame
	R = np.array([[np.cos(theta[jtheta]), -np.sin(theta[jtheta])], [np.sin(theta[jtheta]), np.cos(theta[jtheta])]])

def getGroundIndices(transfAngles, p, z, angles):
	pass

def getGoodIndicesLaserScan(transfAngles, p, z, angles):
	# Indices (of angles or z) that are too close, far, or hit the ground
	indValid = np.logical_and((z < 30), (z > 0.1))
	indGround = getGroundIndices(angles, z)
	return np.logical_and(indValid, indGround)

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
	angles = np.array(np.arange(-135,135.25,0.25)*np.pi/180.)

	# Remove scan points that are too close, far, or hit the ground
	zi = getGoodIndicesLaserScan(transfAngles, p, z, angles)
	z = z[zi]
	angles = angles[zi]

	# Transform z to global frame
	z_occCells = transformLidarH2G(transfAngles, p, z, angles)

	# Use getMapCellsFromRay() to get free cells
	z_freeCells = MU2.getMapCellsFromRay(p[0], p[1], z_occCells[0], z_occCells[1])

	# Update log-odds ratio in m
	m_new = updateMapLogOdds(m, z_occCells, z_freeCells)
	
	return m_new

# - p - Pose of all 'n' particles at time 't'
# - o_t - odometery at time 't' - confirm whether relative and which reference
# - o_t1 - odometery at time 't+1' - confirm whether relative and which reference
# Returns: Pose of all 'n' particles at time 't+1'
def localizationPrediction(p, o_t, o_t1):
	# Get some random noise for each particle

	# p_t+1 = p_t ++ (o_t+1 -- o_t) ++ noise

	pass

# - P - Pose of all 'n' particles at time 't+1'
# - W - Weight of all 'n' particles at time 't'
# - z - represents the laser scan (relative to head frame) at time 't'
# - m - The map at time 't+1'
# transfAngles - (head angles, rpy) used to transform the laser scan from head to global frame
# Returns: w_(t+1) - Weight of all 'n' particles at time 't+1'
def localizationUpdate(P, W, z, m, transfAngles):
	# For each particle
	for i in range(P.shape[0]):

		# Transform z to global frame
		pass

		# Remove scan points that are too close, far, or hit the ground

		# Compute correlation of the scan points with the map

		# Update the weight of the particle

	pass

# - p - Pose of all 'n' particles at time 't+1'
# - w - Weight of all 'n' particles at time 't+1'
# - numParticles - number of particles to return
# Returns: (p, w) - new particles and their weights
def resampleIfNeeded(p, w, numParticles):
	pass

def slam(lidar, joint):
	numParticles = 200
	lidar, joint = getDataAtSameTimestamp(lidar, joint)
	P, W, Map = initSlam(lidar, joint, numParticles)

	for t in xrange(len(lidar) - 1):
		z_headFrame = lidar[t]['scan'][0]
		trans = (joint['head_angles'], joint['rpy'])

		p_mle = getMLEParticle(P, W) 

		Map = mapping(z_headFrame, p_mle, trans, Map)

		P = localizationPrediction(P, lidar[t]['pose'][0], lidar[t+1]['pose'][0])

		W = localizationUpdate(P, W, z_headFrame, Map, trans)

		P, W = resampleIfNeeded(P, W, numParticles)

		showMaps(P, W, Map)

################################################################################################

def runSlam():
	lidarFile = 'Proj4_2018_Train/data/train_lidar0'
	jointFile = 'Proj4_2018_Train/data/train_joint0'

	lidar = ld.get_lidar(lidarFile)
	joint = ld.get_joint(jointFile)

	slam(lidar, joint)

if __name__ == "__main__":
	# replayLidarData()
	# randomExperiment()

	runSlam()	