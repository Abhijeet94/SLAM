import sys
import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import matplotlib.image as mpimg
import time
from scipy.misc import logsumexp
import pickle as pkl

sys.path.insert(0,'Proj4_2018_Train')
sys.path.insert(0,'Proj4_2018_Train/MapUtils')
sys.path.insert(0,'Proj4_2018_Train_rgb')
sys.path.insert(0,'Proj4_2018_Train_rgb/cameraParam')

import load_data as ld
import p4_util as util
import MapUtils as MU
import MapUtils2 as MU2
from utils import *
from slam import *

################################################################################################

def replayDepthData():
	depth0 = ld.get_depth('Proj4_2018_Train_rgb/DEPTH_0')
	# pdb.set_trace()
	util.replay_depth(depth0)

def replayRGBData():
	rgb0 = ld.get_rgb('Proj4_2018_Train_rgb/RGB_3_1')
	pdb.set_trace()
	util.replay_rgb(rgb0)

################################################################################################

def camera2originalFrame(arr):
	R = np.array([[0, 0, 1], \
		[-1, 0, 0], \
		[0, -1, 0]])

	lenx = arr.shape[0]
	leny = arr.shape[1]
	arr = np.matmul(R, arr.reshape(lenx, leny, 3, 1)).reshape(lenx, leny, 3)
	return arr	

def original2cameraFrame(arr):
	R = np.array([[0, -1, 0], \
		[0, 0, -1], \
		[1, 0, 0]])

	lenx = arr.shape[0]
	leny = arr.shape[1]
	arr = np.matmul(R, arr.reshape(lenx, leny, 3, 1)).reshape(lenx, leny, 3)
	return arr

################################################################################################

def transformKinectH2B(constants, depthImage):
	u_len = depthImage.shape[0]
	v_len = depthImage.shape[1]
	depthArray = np.transpose(np.indices((u_len, v_len)), (1, 2, 0))
	depthArray = np.dstack((depthArray, depthImage))

	depthArray[:, :, 0] = depthArray[:, :, 0] - (v_len/2.0)
	depthArray[:, :, 1] = depthArray[:, :, 1] - (u_len/2.0)

	fu = constants['fu_depth']
	fv = constants['fv_depth']

	depthArray_cart = np.zeros(depthArray.shape)
	depthArray_cart[:, :, 0] = np.multiply(depthArray[:, :, 0], depthArray[:, :, 2]) / fu
	depthArray_cart[:, :, 1] = np.multiply(depthArray[:, :, 1], depthArray[:, :, 2]) / fv
	depthArray_cart[:, :, 2] = depthArray[:, :, 2]
	# depthArray_cart now consists of cartesian coordinates in camera frame
	# We need to rotate it to the body frame
	pdb.set_trace()

	depthArray_cart = camera2originalFrame(depthArray_cart)
	depthArray_cart[:, :, 2] = depthArray[:, :, 2] + constants['camera2neck_distance']

	yaw_head = constants['yaw_head']
	pitch_head = constants['pitch_head']

	rotz = np.array([[np.cos(yaw_head), -np.sin(yaw_head), 0], \
		[np.sin(yaw_head), np.cos(yaw_head), 0], \
		[0, 0, 1]])

	roty = np.array([[np.cos(pitch_head), 0, np.sin(pitch_head)], \
		[0, 1, 0], \
		[-np.sin(pitch_head), 0, np.cos(pitch_head)]])

	depthArray_cart_body = np.matmul(np.matmul(rotz, roty), depthArray_cart.reshape(u_len, v_len, 3, 1)).reshape(u_len, v_len, 3)
	depthArray_cart_body[:, :, 2] = depthArray_cart_body[:, :, 2] + constants['neck2ground_distance']
	return depthArray_cart_body

def transformKinectB2G(constants, pose, depthArray_body):
	pose = pose.reshape(3)
	x = pose[0]
	y = pose[1]
	theta = pose[2]
	tB2G = np.array([[np.cos(theta), -np.sin(theta), 0, x], \
		[np.sin(theta), np.cos(theta), 0, y], \
		[0, 0, 1, 0], \
		[0, 0, 0, 1]])

	dabx = depthArray_body.shape[0]
	daby = depthArray_body.shape[1]
	depthArray_body = np.dstack((depthArray_body, np.ones((dabx, daby))))
	depthArray_global = np.matmul(tB2G, depthArray_body.reshape(dabx, daby, 4, 1)).reshape(dabx, daby, 4)
	return depthArray_global[:, :, 0:3]

def transformKinectH2G(constants, pose, depthImage):
	# Transform depthImage to global frame from depth camera frame
	depthArray_body = transformKinectH2B(constants, depthImage)
	depthArray_global = transformKinectB2G(constants, pose, depthArray_body)
	return depthArray_global # this is in original frame (not in camera axis)

################################################################################################

def getRGBforDepthPixels(depthImage, cameraImage, constants):
	u_len_depth = depthImage.shape[0]
	v_len_depth = depthImage.shape[1]
	depthArray = np.transpose(np.indices((u_len_depth, v_len_depth)), (1, 2, 0))
	depthArray = np.dstack((depthArray, depthImage))

	fu = constants['fu_depth']
	fv = constants['fv_depth']

	depthArray_cart = np.zeros(depthArray.shape)
	depthArray_cart[:, :, 0] = np.multiply(depthArray[:, :, 0], depthArray[:, :, 2]) / fu
	depthArray_cart[:, :, 1] = np.multiply(depthArray[:, :, 1], depthArray[:, :, 2]) / fv
	depthArray_cart[:, :, 2] = depthArray[:, :, 2]
	# depthArray_cart now consists of cartesian coordinates in camera frame

	tD2R = np.array([[constants['R'][0, 0], constants['R'][0, 1], constants['R'][0, 2], constants['T'][0, 0]], \
		[constants['R'][1, 0], constants['R'][1, 1], constants['R'][1, 2], constants['T'][1, 0]], \
		[constants['R'][2, 0], constants['R'][2, 1], constants['R'][2, 2], constants['T'][2, 0]], \
		[0, 0, 0, 1]])

	# Adding a layer of 1's for 4d multiplication, then multiplying with Transformation matrix
	dacx = depthArray_cart.shape[0]
	dacy = depthArray_cart.shape[1]
	rgb_cart = np.dstack((depthArray_cart, np.ones((dacx, dacy))))
	rgb_cart = np.matmul(tD2R, rgb_cart.reshape(dacx, dacy, 4, 1))[:, :, 0:3, 0]

	fu = constants['fu_rgb']
	fv = constants['fv_rgb']

	# Convert to camera image plane
	rgbCoord = np.zeros((dacx, dacy, 2))
	rgbCoord[:, :, 0] = np.divide(rgb_cart[:, :, 0], rgb_cart[:, :, 2]) * fu
	rgbCoord[:, :, 1] = np.divide(rgb_cart[:, :, 1], rgb_cart[:, :, 2]) * fv

	# Get colors from the camera image
	rgbCoord = np.rint(rgbCoord)
	rgbCoord = rgbCoord.astype(int)
	validX = np.logical_and(rgbCoord[:, :, 0] < cameraImage.shape[0], rgbCoord[:, :, 0] >= 0)
	validY = np.logical_and(rgbCoord[:, :, 1] < cameraImage.shape[1], rgbCoord[:, :, 1] >= 0)
	vi = np.logical_and(validX, validY)
	colors = np.zeros((dacx, dacy, 3))
	colors[vi, :] = cameraImage[rgbCoord[vi, 0], rgbCoord[vi, 1], :]
	colors[~vi, :] = -1
	return colors

################################################################################################

def showTextureMap(depth, rgb, depthIndex, p_mle, Map, constants, TM):
	depthData = depth[depthIndex]
	rgbData = rgb[depthIndex]

	depthArray_global = transformKinectH2G(constants, p_mle, depthData['depth']/1000.0) # in original axes
	colors = getRGBforDepthPixels(depthData['depth']/1000.0, rgbData['image'], constants)

	vi = depthArray_global[:, :, 2] < 0.025 # ground indices
	print '1) ' + str(np.sum(vi))
	vi = np.logical_and(vi, depthArray_global[:, :, 0] >= Map['xmin'])
	vi = np.logical_and(vi, depthArray_global[:, :, 0] <= Map['xmax'])
	print '2) ' + str(np.sum(vi))
	vi = np.logical_and(vi, depthArray_global[:, :, 1] >= Map['ymin'])
	vi = np.logical_and(vi, depthArray_global[:, :, 1] <= Map['ymax'])
	print '3) ' + str(np.sum(vi))
	vi_colors = colors[:, :, 0] != -1
	print '4) ' + str(np.sum(vi_colors))
	vi = np.logical_and(vi, vi_colors)
	print '5) ' + str(np.sum(vi))

	xis = np.ceil((depthArray_global[vi, 0] - Map['xmin']) / Map['res'] ).astype(np.int16)-1
	yis = np.ceil((depthArray_global[vi, 1] - Map['ymin']) / Map['res'] ).astype(np.int16)-1

	TM[xis, yis, 0] = colors[vi, 0]
	TM[xis, yis, 1] = colors[vi, 1]
	TM[xis, yis, 2] = colors[vi, 2]

	plt.imshow(TM)
	fig.canvas.draw()
	fig.canvas.flush_events()
	fig.show()

def isTMtime(lidar, depth, timestep, depthIndex, step):
	rv = False
	if len(lidar) > timestep + step:
		lidarCurrentTime = lidar[timestep]['t'][0]
		lidarNextTime = lidar[timestep + step]['t'][0]
		depthCurrentTime = depth[depthIndex]['t'][0][0]
		if lidarCurrentTime <= depthCurrentTime and lidarNextTime > depthCurrentTime:
			rv = True
	return rv


################################################################################################
def slamWithTM(lidar, joint, depth, rgb):
	numParticles = 100
	lidar, joint = getDataAtSameTimestamp(lidar, joint)
	P, W, Map = initSlam(lidar, joint, numParticles)
	poseHistoryX = []
	poseHistoryY = []
	step = 5
	startPos =  0#40 * 500

	IRData = pkl.load(open('Proj4_2018_Train_rgb/cameraParam/IRcam_Calib_result.pkl', 'rb'))
	RGBData = pkl.load(open('Proj4_2018_Train_rgb/cameraParam/RGBcamera_Calib_result.pkl', 'rb'))
	ExParams = pkl.load(open('Proj4_2018_Train_rgb/cameraParam/exParams.pkl', 'rb'))

	constants = {}
	constants['neck2ground_distance'] = 1.26
	constants['camera2neck_distance'] = 0.07
	constants['fu_depth'] = IRData['fc'][0]/1000.0
	constants['fv_depth'] = IRData['fc'][1]/1000.0
	constants['fu_rgb'] = RGBData['fc'][0]/1000.0
	constants['fv_rgb'] = RGBData['fc'][1]/1000.0
	constants['cc_depth'] = np.array(IRData['cc'])/1000.0
	constants['cc_rgb'] = np.array(RGBData['cc'])/1000.0
	constants['R'] = ExParams['R']
	constants['T'] = ExParams['T'] / 1000.0

	depthIndex = 0
	prevTM = np.zeros((Map['sizex'], Map['sizey'], 3), 'uint8')
	while depth[depthIndex]['t'][0][0] < lidar[0]['t'][0]:
		depthIndex = depthIndex + 1
	print depthIndex

	for t in xrange(startPos, len(lidar) - step * 2, step):
		z_headFrame = lidar[t]['scan'][0]
		trans = (joint['head_angles'][0][t], joint['head_angles'][1][t], joint['rpy'][0][t], joint['rpy'][1][t], joint['rpy'][2][t])

		p_mle = getMLEParticle(P, W) 
		poseHistoryX.append(np.ceil((p_mle[0] - Map['xmin']) / Map['res']) - 1)
		poseHistoryY.append(np.ceil((p_mle[1] - Map['xmin']) / Map['res']) - 1)
		Map = mapping(z_headFrame, p_mle, trans, Map)
		P = localizationPrediction(P, lidar[t]['pose'][0], lidar[t+step]['pose'][0])
		W = localizationUpdate(P, W, z_headFrame, Map, trans)
		P, W = resampleIfNeeded(P, W, numParticles)

		#########################################################################
		#########################################################################
		#########################################################################

		if t % 500 == 0:
			print 'Iteration: ' + str(t)

		if isTMtime(lidar, depth, t, depthIndex, step):
			constants['yaw_head'] = depth[depthIndex]['head_angles'][0][0]
			constants['pitch_head'] = depth[depthIndex]['head_angles'][0][0]

			showTextureMap(depth, rgb, depthIndex, p_mle, Map, constants, prevTM)
			depthIndex = depthIndex + 1
			print depthIndex

		#########################################################################
		#########################################################################
		#########################################################################

	return P, W, Map, poseHistoryX, poseHistoryY

################################################################################################

def runTextureMap():
	lidarFile = 'Proj4_2018_Train/data/train_lidar0'
	jointFile = 'Proj4_2018_Train/data/train_joint0'
	depthFile = 'Proj4_2018_Train_rgb/DEPTH_0'
	rgbFile = 'Proj4_2018_Train_rgb/RGB_0'

	plt.ion()

	lidar = ld.get_lidar(lidarFile)
	joint = ld.get_joint(jointFile)
	depth = ld.get_depth(depthFile)
	rgb = ld.get_rgb(rgbFile)

	slamWithTM(lidar, joint, depth, rgb)
	print 'Done'
	plt.show(block=True)

if __name__ == "__main__":
	# replayDepthData()
	# replayRGBData()
	runTextureMap()