import sys
import pdb

sys.path.insert(0,'Proj4_2018_Train')
sys.path.insert(0,'Proj4_2018_Train/MapUtils')

import load_data as ld
import p4_util as util

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



################################################################################################
 
# - z - represents the laser scan (relative to head frame) at time 't'
# - p - is the MLE particle at time 't'
# transformation - to transform the laser scan from head to global frame
# - m - The map at time 't'
# Returns: m_(t+1) - The map at time 't+1'
def mapping(z, p, transformation, m):
	pass

# - p - Pose of all 'n' particles at time 't'
# - o_t - odometery at time 't' - confirm whether relative and which reference
# - o_t1 - odometery at time 't+1' - confirm whether relative and which reference
# Returns: Pose of all 'n' particles at time 't+1'
def localizationPrediction(p, o_t, o_t1):
	pass

# - p - Pose of all 'n' particles at time 't+1'
# - w - Weight of all 'n' particles at time 't'
# - z - represents the laser scan (relative to head frame) at time 't'
# - m - The map at time 't+1'
# transformation - to transform the laser scan from head to global frame
# Returns: w_(t+1) - Weight of all 'n' particles at time 't+1'
def localizationUpdate(p, w, z, m, transformation):
	pass

# - p - Pose of all 'n' particles at time 't+1'
# - w - Weight of all 'n' particles at time 't+1'
# Returns: (p, w) - new particles and their weights
def resampleIfNeeded(p, w):
	pass

def slam():
	pass
################################################################################################

if __name__ == "__main__":
	replayLidarData()
	# randomExperiment()