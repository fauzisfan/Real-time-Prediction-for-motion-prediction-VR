# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:23:42 2020

@author: isfan
"""

import numpy as np
import time
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.stats import norm

#from scipy import signal
from sklearn import preprocessing, metrics
from keras import models, layers, regularizers
from scipy import signal
# from disolve import remdiv
# modularized library import
import sys
sys.path.append('/gdrive/My Drive/Colab Notebooks/Motion prediction')

from preparets import preparets
from train_test_split import train_test_split_tdnn
#from calc_future_orientation import calc_future_orientation
from eul2quat_bio import eul2quat_bio
from quat2eul_bio import quat2eul_bio
#from ann_prediction import ann_prediction
from cap_prediction import cap_prediction
from crp_prediction import crp_prediction
from nop_prediction import nop_prediction
from solve_discontinuity import solve_discontinuity
from rms import rms
import quaternion
#from convention_biosignal import convention_biosignal
#from convention_biosignal_quat import convention_biosignal_quat   
import os
import math
# from sklearn.externals import joblib
import joblib

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def calc_optimal_overhead(hmd_orientation, frame_orientation, hmd_projection, pos_norm=0):
        q_d = np.matmul(
                np.linalg.inv(quaternion.as_rotation_matrix(hmd_orientation)),
                quaternion.as_rotation_matrix(frame_orientation)
                )
        #Projection Orientation:
            #hmd_projection[0] : Left (Negative X axis)
            #hmd_projection[1] : Top (Positive Y axis)
            #hmd_projection[2] : Right (Positive X axis)
            #hmd_projection[3] : Bottom (Negative Y axis)
            
        lt = np.matmul(q_d, [hmd_projection[0], hmd_projection[1], 1-pos_norm])
        p_lt = np.dot(np.dot(lt, 1 / lt[2]),1-pos_norm)
        
        rt = np.matmul(q_d, [hmd_projection[2], hmd_projection[1], 1-pos_norm])
        p_rt = np.dot(np.dot(rt, 1 / rt[2]),1-pos_norm)
        
        rb = np.matmul(q_d, [hmd_projection[2], hmd_projection[3], 1-pos_norm])
        p_rb = np.dot(np.dot(rb, 1 / rb[2]),1-pos_norm)
        
        lb = np.matmul(q_d, [hmd_projection[0], hmd_projection[3], 1-pos_norm])
        p_lb =np.dot(np.dot(lb, 1 / lb[2]),1-pos_norm)
        
        p_l = min(p_lt[0], p_rt[0], p_rb[0], p_lb[0])
        p_t = max(p_lt[1], p_rt[1], p_rb[1], p_lb[1])
        p_r = max(p_lt[0], p_rt[0], p_rb[0], p_lb[0])
        p_b = min(p_lt[1], p_rt[1], p_rb[1], p_lb[1])
        
        # p_l = p_l - (-1 - hmd_projection[0])
        # p_t = p_t - (1 - hmd_projection[1])
        # p_r = p_r - (1 - hmd_projection[2])
        # p_b = p_b - (-1 - hmd_projection[3])
        
    # 	size = max(p_r - p_l, p_t - p_b)
    # 	a_overfilling = size * size
        
    # 	a_hmd = (hmd_projection[2] - hmd_projection[0]) * (hmd_projection[1] - hmd_projection[3])
        
        # return [-np.abs(p_l),np.abs(p_t),np.abs(p_r),-np.abs(p_b)]
        margins = np.max(np.abs([p_l, p_t, p_r, p_b]))
        return [-margins, margins, margins, -margins]

def predict_overfilling(predict_orientation, predict_orientation_min, predict_orientation_max, hmd_projection):
	# calc_optimal_overhead(input_orientation, predict_orientation, input_projection)
	q_d_min = np.matmul(
			np.linalg.inv(quaternion.as_rotation_matrix(predict_orientation)),
			quaternion.as_rotation_matrix(predict_orientation_min)
			)
	q_d_max = np.matmul(
			np.linalg.inv(quaternion.as_rotation_matrix(predict_orientation)),
			quaternion.as_rotation_matrix(predict_orientation_max)
			)
	
	lt_min = np.matmul(q_d_min, [hmd_projection[0], hmd_projection[1], 1])
	p_lt_min = np.dot(lt_min, 1 / lt_min[2])
	
	rt_min = np.matmul(q_d_min, [hmd_projection[2], hmd_projection[1], 1])
	p_rt_min = np.dot(rt_min, 1 / rt_min[2])
	
	rb_min = np.matmul(q_d_min, [hmd_projection[2], hmd_projection[3], 1])
	p_rb_min = np.dot(rb_min, 1 / rb_min[2])
	
	lb_min = np.matmul(q_d_min, [hmd_projection[0], hmd_projection[3], 1])
	p_lb_min = np.dot(lb_min, 1 / lb_min[2])
	
	lt_max = np.matmul(q_d_max, [hmd_projection[0], hmd_projection[1], 1])
	p_lt_max = np.dot(lt_max, 1 / lt_max[2])
	
	rt_max = np.matmul(q_d_max, [hmd_projection[2], hmd_projection[1], 1])
	p_rt_max = np.dot(rt_max, 1 / rt_max[2])
	
	rb_max = np.matmul(q_d_max, [hmd_projection[2], hmd_projection[3], 1])
	p_rb_max = np.dot(rb_max, 1 / rb_max[2])
	
	lb_max = np.matmul(q_d_max, [hmd_projection[0], hmd_projection[3], 1])
	p_lb_max = np.dot(lb_max, 1 / lb_max[2])
	
	p_l1 = min(p_lt_min[0], p_rt_min[0], p_rb_min[0], p_lb_min[0])
	p_l2 = min(p_lt_max[0], p_rt_max[0], p_rb_max[0], p_lb_max[0])
	p_t1 = max(p_lt_min[1], p_rt_min[1], p_rb_min[1], p_lb_min[1])
	p_t2 = max(p_lt_max[1], p_rt_max[1], p_rb_max[1], p_lb_max[1])
	p_r1 = max(p_lt_min[0], p_rt_min[0], p_rb_min[0], p_lb_min[0])
	p_r2 = max(p_lt_max[0], p_rt_max[0], p_rb_max[0], p_lb_max[0])
	p_b1 = min(p_lt_min[1], p_rt_min[1], p_rb_min[1], p_lb_min[1])
	p_b2 = min(p_lt_max[1], p_rt_max[1], p_rb_max[1], p_lb_max[1])
	
	p_l = (p_l1+p_l2)/2
	p_t = (p_t1+p_t2)/2
	p_r = (p_r1+p_r2)/2
	p_b = (p_b1+p_b2)/2
	
	
	return [-np.abs(p_l),np.abs(p_t),np.abs(p_r),-np.abs(p_b)]
# 	margins = np.max(np.abs([p_l, p_t, p_r, p_b]))
# 	return [-margins, margins, margins, -margins]

def Robust_overfilling(input_orientation, prediction, input_projection, offset = 1, fixed_param = 1):
    #Get Input Projection Distance for each side in x,y coordinate
    IPDx = input_projection[2]-input_projection[0]
    IPDy = input_projection[1]-input_projection[3]
    	
    	#Define projection distance to the user
    h = 1
    	
    	#Get the corner point distance to the rotation center for each side in x,y coordinate
    rx = np.sqrt(h**2+1/4*IPDx**2)
    ry = np.sqrt(h**2+1/4*IPDy**2)
    	
    	#Get initial input angle to the rotational center
    input_anglex = np.arctan(IPDx/(2*h))
    input_angley = np.arctan(IPDy/(2*h))
    	
    	#Get user's direction based on prediction motion
    pitch_diff = (prediction[0]-input_orientation[0])
    roll_diff = (prediction[1]-input_orientation[1])
    yaw_diff = (prediction[2]-input_orientation[2])
    	
    	#Calculate predicted margin based on translation movement
    x_r = max(input_projection[2],rx*abs(np.sin(input_anglex-yaw_diff)))
    x_l = min(input_projection[0],-rx*abs(np.sin(input_anglex+yaw_diff)))
    y_t = max(input_projection[1],ry*abs(np.sin(input_angley+pitch_diff)))
    y_b = min(input_projection[3],-ry*abs(np.sin(input_angley-pitch_diff)))
    
    	#Calculate predicted margin based on rotational movement
    x_rr = rx*abs(np.sin(input_anglex+abs(roll_diff)))
    x_ll = -rx*abs(np.sin(input_anglex+abs(roll_diff)))
    y_tt = ry*abs(np.sin(input_angley+abs(roll_diff)))
    y_bb = -ry*abs(np.sin(input_angley+abs(roll_diff)))
    	
    	#Calculate final movement
    p_r = x_r+x_rr-IPDx/2
    p_l = x_l+x_ll+IPDx/2
    p_t = y_t+y_tt-IPDy/2
    p_b = y_b+y_bb+IPDy/2
    
    'Enhancement'
    ##	#get largest point
    #	z_yaw = abs(h-rx*np.cos(input_anglex+abs(yaw_diff)))
    #	z_pitch = abs(h-ry*np.cos(input_angley+abs(pitch_diff)))
    #	
    #	if (yaw_diff<0):
    #		p_r = np.sqrt(abs(p_r-p_l)**2+z_yaw**2)+p_l
    #	else: 
    #		p_l = p_r-np.sqrt(abs(p_l-p_r)**2+z_yaw**2)
    #	if (pitch_diff>0):
    #		p_t = np.sqrt(abs(p_t-p_b)**2+z_pitch**2)+p_b
    #	else :
    #		p_b = p_t-np.sqrt(abs(p_b-p_t)**2+z_pitch**2)
    	
    	#Calculate margin based on genrated area
    margin = np.sqrt((p_r-p_l)*(p_t-p_b)*offset)/2
    p_l = -(margin+np.sin(yaw_diff))+(input_projection[0]+1)
    p_t = margin+np.sin(pitch_diff)+(input_projection[1]-1)
    p_r = margin-np.sin(yaw_diff)+(input_projection[2]-1)
    p_b = -(margin-np.sin(pitch_diff))+(input_projection[3]+1)
    	
    'Enhancement ver 2'
    ##	Get dilation on high velocity
    p_r = max(p_r*(np.sin(abs(yaw_diff))*(fixed_param-1)+1),p_r*(np.sin(abs(pitch_diff))*(fixed_param-1)+1))
    p_l = min(p_l*(np.sin(abs(yaw_diff))*(fixed_param-1)+1),p_l*(np.sin(abs(pitch_diff))*(fixed_param-1)+1))
    p_t = max(p_t*(np.sin(abs(pitch_diff))*(fixed_param-1)+1),p_t*(np.sin(abs(yaw_diff))*(fixed_param-1)+1))
    p_b = min(p_b*(np.sin(abs(pitch_diff))*(fixed_param-1)+1), p_b*(np.sin(abs(yaw_diff))*(fixed_param-1)+1))
    	
    
    	#Shifting to predicted point
    #	return [-(abs(p_l)+np.sin(yaw_diff)),abs(p_t)+np.sin(pitch_diff),abs(p_r)-np.sin(yaw_diff),-(abs(p_b)-np.sin(pitch_diff))]
    
    return [-np.abs(p_l),np.abs(p_t),np.abs(p_r),-np.abs(p_b)]
    # margins = np.max(np.abs([p_l, p_t, p_r, p_b]))
    # return [-margins, margins, margins, -margins]
    
def GazeUpdate(proj_pred, proj_input, input_orientation=[0,0,0,0], prediction=[0,0,0,0]):
	overfill_left = proj_pred
	overfill_right = [
		overfill_left[0]-(proj_input[0]+proj_input[2]),
		overfill_left[1],
		overfill_left[2]-(proj_input[0]+proj_input[2]),
		overfill_left[3]
		]
	
	width = overfill_left[2]-overfill_left[0]
	height = overfill_left[1]-overfill_left[3]
	scale = 1#min(width,height)
	
	pitch_diff = (prediction[0]-input_orientation[0])
	yaw_diff = (prediction[2]-input_orientation[2])
	
	left = [
		-((overfill_left[0]+overfill_left[2])/2-np.tan(yaw_diff))/scale,
		-((overfill_left[1]+overfill_left[3])/2+np.tan(pitch_diff))/scale
		]
	
	right = [
		-((overfill_right[0]+overfill_right[2])/2-np.tan(yaw_diff))/scale,
		-((overfill_right[1]+overfill_right[3])/2+np.tan(pitch_diff))/scale
		]
	
	return left, right

def FoveationPatternUpdate(proj_pred, proj_input):
	innerRadii = 0.25
	middleRadii = 0.35
	defaultRadii = 10
	
	videoWidth = 4
	videoHeight = 2
	
	eyeTextureWidth = 0.5
	eyeTextureHeight = 0.25
	
	originalProj = proj_input[1]-proj_input[3]
	overfillWidth = proj_pred[2]-proj_pred[0]
	overfillHeight = proj_pred[1]-proj_pred[3]
	
	widthScale = videoWidth/2/eyeTextureWidth
	heightScale = videoHeight/eyeTextureHeight
	
	encodingSizeWidth = overfillWidth*widthScale
	encodingSizeHeight = overfillHeight*heightScale
	
	videoAspect = videoWidth/2/videoHeight
	
	scale = min(overfillWidth,overfillHeight)/originalProj
	
	Aspect = encodingSizeHeight/encodingSizeWidth*videoAspect
	
	return [innerRadii*scale*Aspect, middleRadii*scale*Aspect, defaultRadii*scale*Aspect], [innerRadii*scale, middleRadii*scale, defaultRadii*scale]

def calCenterRect(proj_data, const = 0.1):
	width = (proj_data[2]-proj_data[0])
	height = (proj_data[1]-proj_data[3])
	size = width*height*const
	return [size]

def calProjection(hmd_orientation, frame_orientation, Radii, center, focus = True):
        q_d = np.matmul(
                np.linalg.inv(quaternion.as_rotation_matrix(hmd_orientation)),
                quaternion.as_rotation_matrix(frame_orientation)
                )
		
        if (focus):
	        cen_min = np.matmul(q_d, [-0.18976885080337524, -0.12733805179595947, 1])
	        center = np.dot(cen_min, 1 / cen_min[2])
		
        width = np.matmul(np.linalg.inv(q_d), [Radii[0]+center[0], center[1], 1])
        p_width = np.dot(np.dot(width, 1 / width[2]),1)
        
        height = np.matmul(np.linalg.inv(q_d), [center[0], Radii[1]+center[1], 1])
        p_height = np.dot(np.dot(height, 1 / height[2]),1)
		
        center = np.matmul(np.linalg.inv(q_d), [center[0], center[1], 1])
        p_center = np.dot(np.dot(center, 1 / center[2]),1)
		
        # p_width = [Radii[0]+center[0], center[1], 1]
        # p_height = [center[0], Radii[1]+center[1], 1]
        # p_center = center
		
        thetaX = np.arctan((p_width[1]-p_center[1])/(p_width[0]-p_center[0]))
        thetaY = np.arctan((p_height[1]-p_center[1])/(p_height[0]-p_center[0]))
		
        if abs(thetaY)<abs(thetaX):
            theta = thetaY
        else:
            theta = thetaX
		
        width = np.sqrt((p_width[1]-p_center[1])**2+(p_width[0]-p_center[0])**2)
        height = np.sqrt((p_height[1]-p_center[1])**2+(p_height[0]-p_center[0])**2)
        
        return np.array([theta, width, height]), p_center

def createPixel(proj_input, number_pixel = 1024):
	difX = (proj_input[2]-proj_input[0])/(number_pixel-1)
	difY = (proj_input[1]-proj_input[3])/(number_pixel-1)
	
	matrix2DX = np.array([proj_input[0]+i*difX for i in range(0,number_pixel)])
	matrix2DY = np.array([proj_input[3]+i*difY for i in range(0,number_pixel)])
	
	return matrix2DX, matrix2DY

def calNumPixel(outer, outer_mid, outer_out,  inner, proj_data, m2DX, m2DY, const = 0.05, num_pixel = 1024):
	total = num_pixel**2
	'''for inner region'''
	angle = outer[0]
	width = outer[1]
	height = outer[2]

	cos_angle = np.cos(angle)
	sin_angle = np.sin(angle)
	
	'''for middle region'''
	angle_mid = outer_mid[0]
	width_mid = outer_mid[1]
	height_mid = outer_mid[2]

	cos_angle_mid = np.cos(angle_mid)
	sin_angle_mid = np.sin(angle_mid)
	
	'''for outer region'''
# 	angle_out = outer_out[0]
# 	width_out = outer_out[1]
# 	height_out = outer_out[2]
# 	
# 	cos_angle_out = np.cos(angle_out)
# 	sin_angle_out = np.sin(angle_out)

	centerX = inner[0]
	centerY = inner[1]
	
	center = [(proj_data[0]+proj_data[2])/2,(proj_data[3]+proj_data[1])/2]
	side = np.sqrt((proj_data[2]-proj_data[0])*(proj_data[1]-proj_data[3])*const)
	
	ellips_pixel = 0
	ellips_pixel_mid = 0
# 	ellips_pixel_out = 0
	rect_pixel = 0
	target = 0
	f = 0
	for i in range (num_pixel):
		for j in range (num_pixel):
			
			ellips = (m2DX[i]*cos_angle+m2DY[j]*sin_angle-centerX)**2/width**2+(m2DY[j]*cos_angle-m2DX[i]*sin_angle-centerY)**2/height**2<=1
			ellips_mid = (m2DX[i]*cos_angle_mid+m2DY[j]*sin_angle_mid-centerX)**2/width_mid**2+(m2DY[j]*cos_angle_mid-m2DX[i]*sin_angle_mid-centerY)**2/height_mid**2<=1
# 			ellips_out = (m2DX[i]*cos_angle_out+m2DY[j]*sin_angle_out-centerX)**2/width_out**2+(m2DY[j]*cos_angle_out-m2DX[i]*sin_angle_out-centerY)**2/height_out**2<=1
			rect = m2DX[i]<=side/2+center[0] and m2DX[i]>=-side/2+center[0] and m2DY[j]<=side/2+center[1] and m2DY[j]>=-side/2+center[1]
			
			
			if (ellips and rect):
				target +=1
				ellips_pixel +=1
				rect_pixel +=1
				f += contrastSensitivity(m2DX[i], m2DY[j], outer, inner, cPoint = center, v = 1, CT = 1)
			elif (ellips):
				ellips_pixel +=1
				f += contrastSensitivity(m2DX[i], m2DY[j], outer, inner, cPoint = center, v = 1, CT = 1)
			elif (ellips_mid and rect):
				ellips_pixel_mid +=1
				rect_pixel +=1
			elif (ellips_mid):
				ellips_pixel_mid +=1
# 			elif (ellips_out and rect):
# 				ellips_pixel_out +=1
# 				rect_pixel +=1
# 			elif (ellips_out):
# 				ellips_pixel_out +=1
			elif (rect):
				rect_pixel +=1
			
	
	return ellips_pixel/total*100, (ellips_pixel-target)/rect_pixel*100, target/rect_pixel*100, ellips_pixel_mid/total*100, (total-(ellips_pixel+ellips_pixel_mid))/total*100, f/ellips_pixel

def contrastSensitivity(matrix2DX, matrix2DY, outer, inner, cPoint = [0,0], v = 1, CT = 1, minCT = 1/64, e2 = 2.3, alpha = 0.106, n = 1024):
	
	''' calculate all foveation region called by calNumPixel function'''
	dist = np.sqrt((matrix2DX-cPoint[0])**2+(matrix2DY-cPoint[1])**2)/v
	e = np.arctan(dist)*180/np.pi
	f = e2*np.log(CT/minCT)/(alpha*(e+e2))

	''' calculate separated all foveation region'''
# 	cPoint = [(matrix2DX[n-1]+matrix2DX[0])/2,(matrix2DY[n-1]+matrix2DY[0])/2]
# 	angle = outer[0]
# 	width = outer[1]
# 	height = outer[2]
# 	
# 	centerX = inner[0]
# 	centerY = inner[1]
# 	
# 	cos_angle = np.cos(angle)
# 	sin_angle = np.sin(angle)
# 	
# 	f = 0
# 	count = 0
# 	for i in range(len(matrix2DX)):
# 		for j in range(len(matrix2DY)):
# 			ellips = (matrix2DX[i]*cos_angle+matrix2DY[j]*sin_angle-centerX)**2/width**2+(matrix2DY[j]*cos_angle-matrix2DX[i]*sin_angle-centerY)**2/height**2<=1
# 			if (ellips):
# 				dist = np.sqrt((matrix2DX[i]-cPoint[0])**2+(matrix2DY[j]-cPoint[1])**2)/v
# 				e = np.arctan(dist)*180/np.pi
# 				f += e2*np.log(CT/minCT)/(alpha*(e+e2))
# 				count +=1
	
	''' calculate only center of foveated region'''
# 	f = np.empty(len(poin), dtype =float)
# 	for i in range(len(poin)):
# 		dist = np.sqrt((poin[i,0]-cPoint[0])**2+(poin[i,1]-cPoint[1])**2)/v
# 		e = np.arctan(dist)*180/np.pi
# 		f[i] = e2*np.log(CT/minCT)/(alpha*(e+e2))

	'''Calculate all the pixels'''
# 	f = np.empty((n,n), dtype =float)
# 	for i in range(len(matrix2DX)):
# 		for j in range(len(matrix2DY)):
# 			dist = np.sqrt((matrix2DX[i]-cPoint[0])**2+(matrix2DY[j]-cPoint[1])**2)/v
# 			e = np.arctan(dist)*180/np.pi
# 			f[i,j] = e2*np.log(CT/minCT)/(alpha*(e+e2))
	
	return f

def plotGraph(outer, inner, cp, axis, const):
	angle = outer[0]
	width = outer[1]
	height = outer[2]
	
	centerX = inner[0]
	centerY = inner[1]
	
	x_ellipse = np.arange(-width,width,0.0001)
	x_ellipse_squared = np.square(x_ellipse)
	
	y_ellipse_plus = (height/width)*np.sqrt(-x_ellipse_squared+width*width)
	y_ellipse_minus = -(height/width)*np.sqrt(-x_ellipse_squared+width*width)
	
	inp = [(axis[0]+axis[2])/2,(axis[3]+axis[1])/2]
	
	side = np.sqrt((axis[2]-axis[0])*(axis[1]-axis[3])*const)
	
	plt.figure()
	
	'''Plot Center Point'''
	plt.scatter(cp[0],cp[1],marker="x")
	plt.legend(['Point Score: {:.2f}'.format((cp[0]*np.cos(angle)+cp[1]*np.sin(angle)-centerX)**2/width**2+
										(cp[1]*np.cos(angle)-cp[0]*np.sin(angle)-centerY)**2/height**2)])
	
	'''plot Center Rectangular'''
	plt.plot(np.full(100,-side/2+inp[0]), np.linspace(inp[1]-side/2,inp[1]+side/2,100),color='blue')
	plt.plot(np.full(100,side/2+inp[0]), np.linspace(inp[1]-side/2,inp[1]+side/2,100), color='blue')
	plt.plot(np.linspace(inp[0]-side/2,inp[0]+side/2,100),np.full(100,-side/2+inp[1]), color='blue')
	plt.plot(np.linspace(inp[0]-side/2,inp[0]+side/2,100),np.full(100,side/2+inp[1]), color='blue')
	
	'''plot Ellipse Region'''
	plt.scatter(centerX + x_ellipse*np.cos(angle)+y_ellipse_plus*np.sin(angle), centerY - x_ellipse*np.sin(angle)+y_ellipse_plus*np.cos(angle), color='red',marker='.', linewidth=0.1)
	plt.scatter(centerX + x_ellipse*np.cos(angle)+y_ellipse_minus*np.sin(angle), centerY - x_ellipse*np.sin(angle)+y_ellipse_minus*np.cos(angle), color='red',marker='.', linewidth=0.1)
	
	plt.xlim([axis[0],axis[2]])
	plt.ylim([axis[3],axis[1]])
	plt.show()

	
anticipation_time = 300
'''
#####   SYSTEM INITIALIZATION    #####
'''
tf.reset_default_graph()

tf.set_random_seed(2)

np.random.seed(2)

scheme = 'model' #(optimal|model|fix|ellips|step)
'''for model'''
offset = 0.75
gain = 1.6

#parser = argparse.ArgumentParser(description='Offline Motion Prediction')
#parser.add_argument('-a', '--anticipation', default=300, type=int)

#args = parser.parse_args()
# infile = "QuestNew_20210719_scene(3)_user(1.1)"
# infile = "QuestNew_20210726_scene(3)_user(1)"
# infile = "QuestNew_20210726_scene(3)_user(2)"
infile = "QuestNew_20210726_scene(3)_user(3)"

try:
    stored_df = pd.read_csv(infile+'_cut.csv')
    train_gyro_data = np.array(stored_df[['input_head_angular_vec_x', 'input_head_angular_vec_y', 'input_head_angular_vec_z']], dtype=np.float)
    train_hand_gyro_data = np.array(stored_df[['input_right_hand_angular_vec_x', 'input_right_hand_angular_vec_y', 'input_right_hand_angular_vec_z']], dtype=np.float)
    train_acce_data = np.array(stored_df[['input_head_acceleration_x', 'input_head_acceleration_y', 'input_head_acceleration_z','input_right_hand_acceleration_x', 'input_right_hand_acceleration_y', 'input_right_hand_acceleration_z']], dtype=np.float)
    train_eule_data = np.array(stored_df[['input_head_orientation_pitch', 'input_head_orientation_roll', 'input_head_orientation_yaw']], dtype=np.float)
    train_hand_eule_data = np.array(stored_df[['input_right_hand_orientation_pitch', 'input_right_hand_orientation_roll', 'input_right_hand_orientation_yaw']], dtype=np.float)
    train_position_data = np.array(stored_df[['input_right_eye_position_x','input_right_eye_position_y','input_right_eye_position_z', 'input_left_eye_position_x','input_left_eye_position_y','input_left_eye_position_z','input_right_hand_position_x','input_right_hand_position_y','input_right_hand_position_z']])
    proj_data = np.array(stored_df[['input_camera_projection_left', 'input_camera_projection_top', 'input_camera_projection_right', 'input_camera_projection_bottom']], dtype=np.float32)
    pred_proj_data = np.array(stored_df[['predicted_camera_projection_left', 'predicted_camera_projection_top', 'predicted_camera_projection_right', 'predicted_camera_projection_bottom']], dtype=np.float32)
    pred_time = np.array(stored_df[['prediction_time']], dtype=np.float32)
    train_time_data = np.array(stored_df['timestamp'], dtype=np.float)
    train_time_data = train_time_data / 705600000
    train_data_id = stored_df.shape[0]
    print('\nSaved data loaded...\n')
    
except:
    raise

# In[6]:


'''
#####   데이터 로드    #####
'''
print('Training data preprocessing is started...')


# Remove zero data from collected training data
system_rate = round((train_data_id+1)/float(np.max(train_time_data) - train_time_data[0]))
idle_period = int(2 * system_rate)
train_gyro_data = train_gyro_data* 180/ np.pi
train_eule_data = train_eule_data * 180 / np.pi
train_hand_eule_data = train_hand_eule_data * 180 / np.pi
train_alfa_data = np.diff(train_gyro_data, axis=0)/np.diff(train_time_data, axis=0).reshape(-1, 1)
train_alfa_data = np.row_stack([np.zeros(shape=(1, train_alfa_data.shape[1]), dtype=np.float), train_alfa_data])
train_alfa_data = train_alfa_data * np.pi / 180


"""Velocity data"""
train_velocity_data = np.diff(train_eule_data, axis=0)/np.diff(train_time_data, axis=0).reshape(-1, 1)
train_velocity_data = np.row_stack([np.zeros(shape=(1, train_velocity_data.shape[1]), dtype=np.float), train_velocity_data])

"""Acceleration diff"""
train_acce_diff_data = np.diff(train_velocity_data, axis =0)/ np.diff(train_time_data, axis=0).reshape(-1, 1)
train_acce_diff_data =np.row_stack([np.zeros(shape=(1, train_acce_diff_data.shape[1]), dtype=np.float), train_acce_diff_data])
# Calculate the head orientation
#train_gyro_data = train_gyro_data * np.pi / 180
train_acce_data = train_acce_data / 9.8

train_eule_data = solve_discontinuity(train_eule_data)
train_hand_eule_data = solve_discontinuity(train_hand_eule_data)


# Create data frame of all features and smoothing
sliding_window_time = 100
sliding_window_size = int(np.round(sliding_window_time * system_rate / 1000))

ann_feature = np.column_stack([train_eule_data,
                               # train_hand_eule_data,
                               train_gyro_data,
                               # train_hand_gyro_data,
                               train_alfa_data,
                               # train_position_data,
#                               train_magn_data,
                               ])
    
feature_name = ['head_pitch', 'head_roll', 'head_yaw',
                # 'right_hand_pitch', 'right_hand_roll', 'right_hand_yaw',
                'head_gX', 'head_gY', 'head_gZ',
                # 'right_hand_gX', 'right_hand_gY', 'right_hand_gZ',
                'head_aX', 'head_aY', 'head_aZ',
                # 'right_eye_x', 'right_eye_y','right_eye_z', 'left_eye_x', 'left_eye_y','left_eye_z', 'right_hand_x', 'right_hand_y','right_hand_z'
                ]

ann_feature_df = pd.DataFrame(ann_feature, columns=feature_name)
#ann_feature_df = ann_feature_df.rolling(sliding_window_size, min_periods=1).mean()


# Create the time-shifted IMU data as the supervisor and assign the ann_feature as input
#anticipation_time = args.anticipation  # 앞 셀에서 정의함
anticipation_size = int(np.round(anticipation_time * system_rate / 1000))
print('anticipation size = ', anticipation_size)

#lhood1 = 100
#lhood2 = 200
#lhood3 = 300
#lhood1_size = int(np.round(lhood1 * system_rate / 1000))
#lhood2_size = int(np.round(lhood2 * system_rate / 1000))
#lhood3_size = int(np.round(lhood3 * system_rate / 1000))

spv_name = ['head_pitch', 'head_roll', 'head_yaw',
            # 'right_hand_pitch', 'right_hand_roll', 'right_hand_yaw',
            # 'right_eye_x', 'right_eye_y','right_eye_z', 'left_eye_x', 'left_eye_y','left_eye_z', 'right_hand_x', 'right_hand_y','right_hand_z',
            ]
target_series_df = ann_feature_df[spv_name].iloc[anticipation_size::].reset_index(drop=True)
input_series_df = ann_feature_df.iloc[:-anticipation_size].reset_index(drop=True)

input_nm = len(input_series_df.columns)
target_nm = len(target_series_df.columns)


# In[17]:


'''
#####   NN 입력 데이터 준비    #####
'''

# Neural network parameters
DELAY_SIZE = int(100 * (system_rate / 1000))  # 어떤 용도? 샘플 윈도우?

print("DELAY_SIZE =", DELAY_SIZE)

# Variables
# TRAINED_MODEL_NAME = './QuestNew_realtime_model_20210719'
# TRAINED_MODEL_NAME = './QuestNew_realtime_model_20210726_1'
# TRAINED_MODEL_NAME = './QuestNew_realtime_model_20210726_2'
TRAINED_MODEL_NAME = './QuestNew_realtime_model_20210726_3'


# Import datasets
input_series = np.array(input_series_df)
target_series = np.array(target_series_df)


""""""""" New Preprocessong """""""""
## Split training and testing data
#x_seq, t_seq = preparets(input_series, target_series, DELAY_SIZE)
#data_length = x_seq.shape[0]
#scaler = preprocessing.StandardScaler().fit(training_series)	# fit saves normalization coefficient into scaler
#
#x_seq, t_seq = remdiv(x_seq, t_seq, DELAY_SIZE)
#
##Normalize training data, then save the normalization coefficient
#for i in range(0,len(x_seq)):
#    x_seq[i,:,:] = scaler.transform(x_seq[i,:,:])
#    
#x_train, t_train, x_test, t_test = train_test_split_tdnn(x_seq, t_seq, TEST_SIZE)

""""""""" Old Preprocessong """""""""
# Save it
# scaler_file = "QuestNew_realtime_scaller_20210719.save"
# scaler_file = "QuestNew_realtime_scaller_20210726_1.save"
# scaler_file = "QuestNew_realtime_scaller_20210726_2.save"
scaler_file = "QuestNew_realtime_scaller_20210726_3.save"
#
## Load it 
tempNorm = joblib.load(scaler_file)

#Get normalized based data on the 
# normalizer = preprocessing.StandardScaler()
# tempNorm = normalizer.fit(np.split(input_series,2)[0])
# joblib.dump(tempNorm, scaler_file) 

#Normalizer used on input series
input_norm = tempNorm.transform(input_series)


# Reformat the input into TDNN format
x_seq, t_seq = preparets(input_norm, target_series, DELAY_SIZE)
data_length = x_seq.shape[0]
print('Anticipation time: {}ms\n'.format(anticipation_time))

# x_train, t_train, x_test, t_test = train_test_split_tdnn(x_seq, t_seq, 0.5)

# Reset the whole tensorflow graph
tf.reset_default_graph()


# Set up the placeholder to hold inputs and targets
x = tf.placeholder(dtype=tf.float32, shape=(None, DELAY_SIZE, input_nm))
t = tf.placeholder(dtype=tf.float32)


# In[20]:


# Define TDNN model
model = models.Sequential([
        layers.InputLayer(input_tensor=x, input_shape=(DELAY_SIZE, input_nm)),
        layers.Conv1D(27, DELAY_SIZE, activation=tf.nn.relu, input_shape=(DELAY_SIZE, input_nm), use_bias=True, kernel_regularizer=regularizers.l2(0.01)),
        layers.Flatten(),
        layers.Dense(9, activation=tf.nn.relu, use_bias=True, kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.2),
        layers.Dense(target_nm, activation='linear', use_bias=True, kernel_regularizer=regularizers.l2(0.01)),
        ])

# Get the output of the neural network model
y = model(x)

# Define the loss
loss = tf.reduce_mean(tf.square(t-y))

total_error = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y, t)))
R_squared = tf.subtract(1.0, tf.div(unexplained_error, total_error))

# # Set up the optimizer
# optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='CG', options={'maxiter': 1000, 'gtol': 1e-5})
    
#i = 0
#def callback1(loss):
#    global i
##    i = 0
##    print(', loss:', loss)
#    print('Loss evaluation #', i, ', loss:', loss)
#    i += 1
n_epochs=1000
learning_rate = 0.01
optimizer =  tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

#logdir=os.path.join('tensorboard')
SG_window_size = 5
buff_window = math.ceil(SG_window_size/2)

'''Training Part'''
# window = 10
# start_time = time.time()

# with tf.Session() as sess:
#     # Initiate variable values
#     sess.run(tf.global_variables_initializer())
    
#     # Minimize the error
# #    optimizer.minimize(
# #            session=sess, 
# #            feed_dict={x:x_train, t:t_train},
# #            )
# #    file_writer = tf.summary.FileWriter('logdir', sess.graph)
#     tf.summary.scalar("loss", loss)
#     tf.summary.scalar("accuracy", R_squared)
#     merged_summary = tf.summary.merge_all()
    
# #    for i in range(int(len(t_train)/window)):
#     for epoch in range (n_epochs):
#         summary,_= sess.run([merged_summary,training_op], feed_dict={x:x_train, t:t_train})
# #        print("epoch", epoch)
# #        file_writer.add_summary(summary, epoch)
        
      
#     # Calculate the testing part result
# #    file_writer.close()
#     y_test = sess.run(y, feed_dict={x:x_test})
#     test_mse  = metrics.mean_squared_error(t_test, y_test, multioutput='raw_values')
    
#     print('Test result: {}\n'.format(test_mse))
#     model.save_weights(TRAINED_MODEL_NAME)
    
# fin_time = time.time()
# training_time = fin_time - start_time

'''Testing Part'''
# Evaluate performance of a fully trained network
with tf.Session() as new_sess:    
    # Load best trained network
    model.load_weights(TRAINED_MODEL_NAME)
    

    # Evaluate the network with whole sequence
    y_out = new_sess.run(y, feed_dict={x:x_seq})
#    model.summary()
    # Gyro data format: 
    # [0]: Pitch
    # [1]: Roll
    # [2]: Yaw

# Smoothing with Savgol filtering
#y_out = signal.savgol_filter(y_out, SG_window_size, 3, axis=0)

"""Predict orientation"""
#Nop (defined as input euler shifted for delay size + anticipation size)

# Show some velocity prediction plots
#time_shifted = train_time_data[:-(DELAY_SIZE + anticipation_size)]
# plt.figure()
# plt.plot(time_shifted, y_out[:, 0], linewidth=1)
# plt.plot(time_shifted, t_seq[:, 0], linewidth=1)
# plt.title('Orientation Prediction - SciPy CG - {} ms'.format(np.round(anticipation_time)))
# plt.grid()
# plt.legend(['Predicted', 'Actual'])
# plt.xlabel('Time (s)')
# plt.ylabel('Orientation (deg)')
# plt.show(block=False)

'''
#####   OCULUS PREDICTION COMPARISON   #####
'''
# Recalibrate and align current time head orientation
euler_o = train_eule_data[DELAY_SIZE:-anticipation_size]
hand_euler_o = train_hand_eule_data[DELAY_SIZE:-anticipation_size]
position_o = train_position_data[DELAY_SIZE:-anticipation_size]

# Align current head velocity and acceleration
gyro_o = train_gyro_data[DELAY_SIZE:-anticipation_size]
alfa_o = train_alfa_data[DELAY_SIZE:-anticipation_size]
accel_o = train_acce_data[DELAY_SIZE:-anticipation_size]
velocity_o = train_velocity_data[:-anticipation_size]

# Predict orientation
euler_pred_ann = y_out
euler_pred_cap = cap_prediction(euler_o, gyro_o* np.pi / 180, alfa_o, anticipation_time)
euler_pred_crp = crp_prediction(euler_o, gyro_o* np.pi / 180, anticipation_time)
euler_pred_nop = nop_prediction(euler_o, anticipation_time)

# Calculate prediction error
# Error is defined as difference between:
# Predicted head orientation
# Actual head orientation = Current head orientation shifted by s time
euler_ann_err = np.abs(euler_pred_ann[:-anticipation_size,0:3] - euler_o[anticipation_size:])
euler_cap_err = np.abs(euler_pred_cap[:-anticipation_size] - euler_o[anticipation_size:])
euler_crp_err = np.abs(euler_pred_crp[:-anticipation_size] - euler_o[anticipation_size:])
euler_nop_err = np.abs(euler_pred_nop[:-anticipation_size] - euler_o[anticipation_size:])
# hand_euler_ann_err = np.abs(euler_pred_ann[:-anticipation_size,3:6] - hand_euler_o[anticipation_size:])
# position_ann_err = np.abs(euler_pred_ann[:-anticipation_size,6::] - position_o[anticipation_size:])

''' Plot '''
timestamp_plot = train_time_data[DELAY_SIZE:-2*anticipation_size]
time_offset = timestamp_plot[0]
timestamp_plot = np.array(timestamp_plot)-time_offset

# plt.figure()
# plt.plot(timestamp_plot, euler_pred_ann[:-anticipation_size, 2], linewidth=1,color='magenta')
# # plt.plot(timestamp_plot, ann_pred_rt[:,2], linewidth=1,color='red')
# plt.plot(timestamp_plot, euler_pred_cap[:-anticipation_size, 2], linewidth=1,color='green')
# plt.plot(timestamp_plot, euler_pred_crp[:-anticipation_size, 2], linewidth=1,color='blue')
# plt.plot(timestamp_plot, euler_pred_nop[:-anticipation_size, 2], linewidth=1)
# plt.plot(timestamp_plot, euler_o[anticipation_size:, 2], linewidth=1, color='navy')
# plt.legend(['ANN', 'CAP', 'CRP', 'NOP', 'Actual'])
# plt.title('Orientation Prediction (Yaw)')
# plt.grid()
# plt.xlabel('Time (s)')
# plt.ylabel('Orientation (deg)')
# plt.show(block=False)

# plt.figure()
# plt.plot(timestamp_plot, euler_pred_ann[:-anticipation_size, 1], linewidth=1,color='magenta')
# # plt.plot(timestamp_plot, ann_pred_rt[:,1], linewidth=1,color='red')
# plt.plot(timestamp_plot, euler_pred_cap[:-anticipation_size, 1], linewidth=1,color='green')
# plt.plot(timestamp_plot, euler_pred_crp[:-anticipation_size, 1], linewidth=1,color='blue')
# plt.plot(timestamp_plot, euler_pred_nop[:-anticipation_size, 1], linewidth=1)
# plt.plot(timestamp_plot, euler_o[anticipation_size:, 1], linewidth=1, color='navy')
# plt.legend(['ANN', 'CAP', 'CRP', 'NOP', 'Actual'])
# plt.title('Orientation Prediction (Roll)')
# plt.grid()
# plt.xlabel('Time (s)')
# plt.ylabel('Orientation (deg)')
# plt.show(block=False)

# plt.figure()
# plt.plot(timestamp_plot, euler_pred_ann[:-anticipation_size, 0], linewidth=1,color='magenta')
# # plt.plot(timestamp_plot, ann_pred_rt[:,0], linewidth=1,color='red')
# plt.plot(timestamp_plot, euler_pred_cap[:-anticipation_size, 0], linewidth=1,color='green')
# plt.plot(timestamp_plot, euler_pred_crp[:-anticipation_size, 0], linewidth=1,color='blue')
# plt.plot(timestamp_plot, euler_pred_nop[:-anticipation_size, 0], linewidth=1)
# plt.plot(timestamp_plot, euler_o[anticipation_size:, 0], linewidth=1, color='navy')
# plt.legend(['ANN', 'CAP', 'CRP', 'NOP', 'Actual'])
# plt.title('Orientation Prediction (Pitch)')
# plt.grid()
# plt.xlabel('Time (s)')
# plt.ylabel('Orientation (deg)')
# plt.show(block=False)

'''Hand orientation'''
# plt.figure()
# plt.plot(timestamp_plot, euler_pred_ann[:-anticipation_size, 5], linewidth=1,color='magenta')
# #plt.plot(timestamp_plot, euler_pred_cap[:-anticipation_size, 2], linewidth=1,color='green')
# #plt.plot(timestamp_plot, euler_pred_crp[:-anticipation_size, 2], linewidth=1,color='blue')
# #plt.plot(timestamp_plot, euler_pred_nop[:-anticipation_size, 2], linewidth=1)
# plt.plot(timestamp_plot, hand_euler_o[anticipation_size:, 2], linewidth=1, color='navy')
# plt.legend(['ANN','Actual'])
# plt.title('Right Hand Orientation Prediction (Yaw)')
# plt.grid()
# plt.xlabel('Time (s)')
# plt.ylabel('Orientation (deg)')
# plt.show(block=False)

# plt.figure()
# plt.plot(timestamp_plot, euler_pred_ann[:-anticipation_size, 4], linewidth=1,color='magenta')
# #plt.plot(timestamp_plot, euler_pred_cap[:-anticipation_size, 2], linewidth=1,color='green')
# #plt.plot(timestamp_plot, euler_pred_crp[:-anticipation_size, 2], linewidth=1,color='blue')
# #plt.plot(timestamp_plot, euler_pred_nop[:-anticipation_size, 2], linewidth=1)
# plt.plot(timestamp_plot, hand_euler_o[anticipation_size:, 1], linewidth=1, color='navy')
# plt.legend(['ANN','Actual'])
# plt.title('Right Hand Orientation Prediction (Roll)')
# plt.grid()
# plt.xlabel('Time (s)')
# plt.ylabel('Orientation (deg)')
# plt.show(block=False)

# plt.figure()
# plt.plot(timestamp_plot, euler_pred_ann[:-anticipation_size, 3], linewidth=1,color='magenta')
# #plt.plot(timestamp_plot, euler_pred_cap[:-anticipation_size, 2], linewidth=1,color='green')
# #plt.plot(timestamp_plot, euler_pred_crp[:-anticipation_size, 2], linewidth=1,color='blue')
# #plt.plot(timestamp_plot, euler_pred_nop[:-anticipation_size, 2], linewidth=1)
# plt.plot(timestamp_plot, hand_euler_o[anticipation_size:, 0], linewidth=1, color='navy')
# plt.legend(['ANN','Actual'])
# plt.title('Right Hand Orientation Prediction (Pitch)')
# plt.grid()
# plt.xlabel('Time (s)')
# plt.ylabel('Orientation (deg)')
# plt.show(block=False)

'''Position'''
#plt.figure()
#plt.plot(timestamp_plot, euler_pred_ann[:-anticipation_size, 6], linewidth=1,color='magenta')
##plt.plot(timestamp_plot, euler_pred_cap[:-anticipation_size, 2], linewidth=1,color='green')
##plt.plot(timestamp_plot, euler_pred_crp[:-anticipation_size, 2], linewidth=1,color='blue')
##plt.plot(timestamp_plot, euler_pred_nop[:-anticipation_size, 2], linewidth=1)
#plt.plot(timestamp_plot, position_o[anticipation_size:, 0], linewidth=1, color='navy')
#plt.legend(['ANN','Actual'])
#plt.title('Right Eye Position Prediction (x)')
#plt.grid()
#plt.xlabel('Time (s)')
#plt.ylabel('Position (m)')
#plt.show(block=False)
#
#plt.figure()
#plt.plot(timestamp_plot, euler_pred_ann[:-anticipation_size, 7], linewidth=1,color='magenta')
##plt.plot(timestamp_plot, euler_pred_cap[:-anticipation_size, 2], linewidth=1,color='green')
##plt.plot(timestamp_plot, euler_pred_crp[:-anticipation_size, 2], linewidth=1,color='blue')
##plt.plot(timestamp_plot, euler_pred_nop[:-anticipation_size, 2], linewidth=1)
#plt.plot(timestamp_plot, position_o[anticipation_size:, 1], linewidth=1, color='navy')
#plt.legend(['ANN','Actual'])
#plt.title('Right Eye Position Prediction (y)')
#plt.grid()
#plt.xlabel('Time (s)')
#plt.ylabel('Position (m)')
#plt.show(block=False)
#
#plt.figure()
#plt.plot(timestamp_plot, euler_pred_ann[:-anticipation_size, 8], linewidth=1,color='magenta')
##plt.plot(timestamp_plot, euler_pred_cap[:-anticipation_size, 2], linewidth=1,color='green')
##plt.plot(timestamp_plot, euler_pred_crp[:-anticipation_size, 2], linewidth=1,color='blue')
##plt.plot(timestamp_plot, euler_pred_nop[:-anticipation_size, 2], linewidth=1)
#plt.plot(timestamp_plot, position_o[anticipation_size:, 2], linewidth=1, color='navy')
#plt.legend(['ANN','Actual'])
#plt.title('Right Eye Position Prediction (z)')
#plt.grid()
#plt.xlabel('Time (s)')
#plt.ylabel('Position (m)')
#plt.show(block=False)

"""Online Prediction"""

#Initialize temporary array

ann_pred_rt = np.zeros((0,3), dtype= float)
ann_dummy_rt = np.zeros((0,3), dtype= float)
cap_pred_rt = np.zeros((0,3), dtype= float)
crp_pred_rt = np.zeros((0,3), dtype= float)
nop_pred_rt = np.zeros((0,3), dtype= float)
overfilling_rt = np.zeros((0,4), dtype= float)
gaze_pos = np.zeros((0,2), dtype= float)
radiiPattern = np.zeros((0,6), dtype= float)
error_rt = np.zeros((0,3))
inner = np.zeros((0,3), dtype = float)
outer = np.zeros((0,3), dtype = float)
inner_mid = np.zeros((0,3), dtype = float)
outer_mid = np.zeros((0,3), dtype = float)
inner_out = np.zeros((0,3), dtype = float)
outer_out = np.zeros((0,3), dtype = float)

k = 0
m = 0

# import RealTimePlot as rtp

graphShow = "movement"
coordinate = "roll"

x_seq_rt = np.zeros(shape=(DELAY_SIZE, input_nm))
xd_seq_rt = np.zeros(shape=(DELAY_SIZE, input_nm))
timestamp_rt = np.zeros(shape=(2, 1), dtype= float)
gyro_rt = np.zeros(shape=(2, 3), dtype= float)
rt_counter = 0
y_plot = np.zeros(5)
midVel = np.nanpercentile(np.abs(velocity_o),50, axis = 0)
avgVel = np.nanmean(np.abs(velocity_o), axis=0)
idx = [0,3,1,2]#wxyz

zero_pos = np.nanmean((train_position_data[:,2]+train_position_data[:,5])/2)
head_pos = (train_position_data[:,0:3]+train_position_data[:,3:6])/2

'''For Step-Wise'''
err_win=50
tracked_std = np.zeros((0,3), dtype=float)
tracked_mean = np.zeros((0,3), dtype=float)
ellips_values = np.empty((0,4), dtype=np.float32)
ido_values = np.empty((0,4), dtype=np.float32)
pred_overhead = np.empty((0,1), dtype=np.float32)

# [Left, Top, Right, Bottom]
I = np.zeros(4, dtype = np.float32)
D = np.zeros(4, dtype = np.float32)

max_count = 10
icr = 0.05

d99 = np.sqrt(-2*np.log(1-0.99))
tracked_err = np.zeros(shape=(err_win, 3))
rt_tracked = 0

mtp_delay = int(16 * (system_rate / 1000))

with tf.Session() as new_sess:    
	model.load_weights(TRAINED_MODEL_NAME)
	for i in range(0,(len(train_eule_data)- (2*anticipation_size))):
		#Get euler, gyro, and alfa one by one
		nowTime = i + anticipation_size
		velocity_onedata = velocity_o[i].reshape(1,-1)
		
		euler_pred_onedata = solve_discontinuity(train_eule_data[i].reshape(1,-1))
		euler_future_onedata = solve_discontinuity(train_eule_data[i+anticipation_size].reshape(1,-1))
# 		hand_pred_onedata = solve_discontinuity(train_hand_eule_data[i].reshape(1,-1))
		gyro_pred_onedata = train_gyro_data[i].reshape(1,-1)
		hand_gyro_pred_onedata = train_hand_gyro_data[i].reshape(1,-1)
# 		position_pred_onedata = train_position_data[i].reshape(1,-1)
#		alfa_pred_onedata = train_alfa_data[i].reshape(1,-1)
		timestamp_rt[:-1] =timestamp_rt[1:]
		timestamp_rt[-1] = train_time_data[i]
		gyro_rt[:-1] =gyro_rt[1:]
		gyro_rt[-1] = train_gyro_data[i]
		if (i==0):
			alfa_pred_onedata = np.array(np.zeros(shape=(1, 3), dtype=np.float))
		else:
			alfa_pred_onedata = (np.diff(gyro_rt,axis=0)/np.diff(timestamp_rt, axis=0).reshape(-1, 1))*np.pi/180

		timestamp_plot_onedata = i/60 #timestamp_plot[i]

####################################Without SG Filter
		# Gather data until minimum delay is fulfilled
		if rt_counter < DELAY_SIZE:
			temp_rt = np.column_stack([euler_pred_onedata,
                              # hand_pred_onedata,
                              gyro_pred_onedata,
                              # hand_gyro_pred_onedata,
                              alfa_pred_onedata,
                              # position_pred_onedata
                              ])
#			temp_rt = tempNorm.transform(temp_rt)
			x_seq_rt[:-1] = x_seq_rt[1:]
			x_seq_rt[-1] = temp_rt
			
#			if (rt_counter ==sliding_window_size):
#				seq_df = pd.DataFrame(x_seq_rt)
#				seq_df = seq_df.rolling(sliding_window_size, min_periods=1).mean()
#				temp_seq_rt = np.array(seq_df)
#				xd_seq_rt[:-1] = xd_seq_rt[1:]
#				xd_seq_rt[-1] = temp_seq_rt[-1]
#			else:
#				xd_seq_rt[:-1] = xd_seq_rt[1:]
#				xd_seq_rt[-1] = temp_rt

			rt_counter+=1
			continue			
		else:
			#Get temp cap crp and nop
			tempCap = cap_prediction(euler_pred_onedata, gyro_pred_onedata* np.pi / 180, alfa_pred_onedata, anticipation_time)
			tempCrp = crp_prediction(euler_pred_onedata, gyro_pred_onedata* np.pi / 180, anticipation_time)
			tempNop = nop_prediction(euler_pred_onedata, anticipation_time)
			
			cap_pred_rt = np.concatenate((cap_pred_rt, tempCap), axis = 0)
			crp_pred_rt = np.concatenate((crp_pred_rt, tempCrp), axis = 0)
			nop_pred_rt = np.concatenate((nop_pred_rt, tempNop), axis = 0)

			xdd_seq_rt = tempNorm.transform(x_seq_rt)
			y_sample = new_sess.run(y, feed_dict={x:xdd_seq_rt.reshape(1,DELAY_SIZE,input_nm)})
            
##	        Switch to NOP if velocity under 8deg/s
#			for j in range(0,3):
#				if (np.abs(velocity_onedata[:,j])<1.55):
#					y_sample[:,j] = tempNop[:,j]

            #SoftSwitching
			for j in range(0,3):
				xin = (np.abs(velocity_onedata[:,j])-midVel[j])/avgVel[j]
				alfa = sigmoid(xin)
				y_sample[:,j] = alfa*y_sample[:,j] + (1-alfa)*tempNop[:,j]

			ann_pred_rt = np.concatenate((ann_pred_rt, y_sample), axis =0)
			
			diff = (y_sample - euler_pred_onedata)*np.pi/180
			
			if (scheme == "ellips" or scheme == "step"):
				'''Ellipsoidal Overfilling'''
				if rt_counter >= DELAY_SIZE+anticipation_size:
					if rt_tracked < err_win:
						tracked_err[:-1] = tracked_err[1:]
						tracked_err[-1] = ann_pred_rt[i-anticipation_size]-train_eule_data[i].reshape(1,-1)
	                    
						tracked_std = np.concatenate((tracked_std, [np.std(tracked_err[err_win-1-rt_tracked::], axis=0)]), axis = 0)
						tracked_mean = np.concatenate((tracked_mean, [np.mean(tracked_err[err_win-1-rt_tracked::], axis=0)]), axis = 0)
	                    
						rt_tracked += 1
					else:
						tracked_err[:-1] = tracked_err[1:]
						tracked_err[-1] = ann_pred_rt[i-anticipation_size]-train_eule_data[i].reshape(1,-1)
	                    
						tracked_std  = np.concatenate((tracked_std, [np.std(tracked_err, axis=0)]), axis = 0)
						tracked_mean = np.concatenate((tracked_mean, [np.mean(tracked_err, axis=0)]), axis = 0)
	
					ellipsoid_radius_99 = np.std(tracked_err, axis=0)*d99
					
					euler_pred_min = y_sample - ellipsoid_radius_99
					euler_pred_max = y_sample + ellipsoid_radius_99
	                
					quat_quat_data = eul2quat_bio(euler_pred_onedata)
					quat_quat_predict_max = eul2quat_bio(euler_pred_max)
					quat_quat_predict_min = eul2quat_bio(euler_pred_min)
	
					hmd_orientation = quat_quat_data[:,idx]
					hmd_max = quat_quat_predict_max[:, idx]
					hmd_min = quat_quat_predict_min[:, idx]
	                
					input_orientation = np.quaternion(
	                    hmd_orientation[:,0],
	                    hmd_orientation[:,1],
	                    hmd_orientation[:,2],
	                    hmd_orientation[:,3]
	                )
	                
					input_quat_max = np.quaternion(
	                    hmd_max[:,0],
	                    hmd_max[:,1],
	                    hmd_max[:,2],
	                    hmd_max[:,3]
	                    )
	
					input_quat_min = np.quaternion(
	                    hmd_min[:,0],
	                    hmd_min[:,1],
	                    hmd_min[:,2],
	                    hmd_min[:,3]
	                    )
	                
					pred_overhead_values = predict_overfilling(input_orientation, input_quat_min, input_quat_max, proj_data[i])
					
					pred_overhead_values = np.array([pred_overhead_values[0]-np.sin(diff[0,2]),pred_overhead_values[1]+np.sin(diff[0,0]),pred_overhead_values[2]-np.sin(diff[0,2]),pred_overhead_values[3]+np.sin(diff[0,0])])
					
					ellips_values = np.vstack((ellips_values,pred_overhead_values))
	
					if (scheme == "step"):
						#IDO Mechanism
						for m in range (4):
							if ((abs(proj_data[i,m])-abs(pred_overhead_values[m]))>0):# Get error margin
								I[m] += 1
								if (I[m] > max_count): # Increase Margin
									if (m == 0 or m == 3):# Left and Bottom
										pred_overhead_values[m] -= icr
									else: # Top and Right
										pred_overhead_values[m] += icr
				#					I[m] = 0
								D[m] = 0
							else:
								D[m] += 1
								if (D[m] > max_count): # Decrease margin
									if (m == 0 or m == 3):# Left and Bottom
										pred_overhead_values[m] += icr
									else: # Top and Right
										pred_overhead_values[m] -= icr
				#					D[m] = 0
								I[m] = 0
			    
							# Clip on Maximum Value
							if (pred_overhead_values[m] < proj_data[i,m]-0.2):
									pred_overhead_values[m] = proj_data[i,m]-0.2
							elif (pred_overhead_values[m] > proj_data[i,m]+0.2):
									pred_overhead_values[m] = proj_data[i,m]+0.2
						
						ido_values = np.vstack((ido_values,pred_overhead_values))
					
					margin = pred_overhead_values
				else:
					margin = proj_data[i]
			
			elif scheme == 'optimal':
				'''Optimal Overfilling'''
				quat_quat_data = eul2quat_bio(euler_pred_onedata)
				quat_quat_predict = eul2quat_bio(euler_future_onedata)
	         
				hmd_orientation = quat_quat_data[:,idx]
				frame_orientation = quat_quat_predict[:,idx]
	           
				input_orientation = np.quaternion(
	                hmd_orientation[:,0],
	                hmd_orientation[:,1],
	                hmd_orientation[:,2],
	                hmd_orientation[:,3]
	                )
				predict_orientation = np.quaternion(
	                frame_orientation[:,0],
	                frame_orientation[:,1],
	                frame_orientation[:,2],
	                frame_orientation[:,3]
	                )
				pos_norm = head_pos[i,2]-zero_pos

				margin = calc_optimal_overhead(input_orientation, predict_orientation, proj_data[i], pos_norm = pos_norm)

			elif scheme == 'model':
				'''Model-Based Overfilling'''
				margin = Robust_overfilling(euler_pred_onedata[0]*np.pi/180, y_sample[0]*np.pi/180, proj_data[i], offset = offset, fixed_param = gain)#former = 1.1, 1,3
            
			elif scheme == 'fix':
				'''Fixed Overfilling'''
				optimum_overfill_size = (proj_data[i,2]-proj_data[i,0])*(proj_data[i,1]-proj_data[i,3])
				margin_fix = np.sqrt(2*optimum_overfill_size)/2
				margin = np.array([-(margin_fix+np.sin(diff[0,2])),margin_fix+np.sin(diff[0,0]),margin_fix-np.sin(diff[0,2]),-(margin_fix-np.sin(diff[0,0]))])
			
			overfilling_rt = np.concatenate((overfilling_rt, np.reshape(margin, (1, -1))), axis = 0)

# 			right, left = GazeUpdate(margin, proj_data[i],euler_pred_onedata[0]*np.pi/180, y_sample[0]*np.pi/180)
			_, left = GazeUpdate(margin, proj_data[i])
			
# 			size = calCenterRect(proj_data[i], const = 0.1)
			
# 			center_size = np.concatenate((center_size, size), axis = 0)
			
			gaze_pos = np.concatenate((gaze_pos, np.reshape(left, (1,-1))),axis=0)
			
			currPatternX, currPatternY = FoveationPatternUpdate(margin, proj_data[i])
			radiiPattern = np.concatenate((radiiPattern, np.reshape(currPatternX+currPatternY, (1,-1))),axis=0)
			
			'''For Projection'''
			if (rt_counter>=DELAY_SIZE+anticipation_size):
				euler_prev_onedata = solve_discontinuity(train_eule_data[i-mtp_delay].reshape(1,-1))
				pred_onedata = ann_pred_rt[i-rt_counter].reshape(1,-1)
				
				quat_quat_prev = eul2quat_bio(euler_prev_onedata)
				quat_quat_now = eul2quat_bio(pred_onedata)
	         
				hmd_orientation = quat_quat_prev[:,idx]
				frame_orientation = quat_quat_now[:,idx]
	           
				motion_ = np.quaternion(
	                hmd_orientation[:,0],
	                hmd_orientation[:,1],
	                hmd_orientation[:,2],
	                hmd_orientation[:,3]
	                )
				photon_ = np.quaternion(
	                frame_orientation[:,0],
	                frame_orientation[:,1],
	                frame_orientation[:,2],
	                frame_orientation[:,3]
	                )
				
				'''inner Area'''
				temp1, temp2 = calProjection(motion_, photon_, [currPatternX[0],currPatternY[0]], center = left, focus = True)
				outer = np.concatenate((outer, np.reshape(temp1, (1,-1))),axis=0)
				inner = np.concatenate((inner, np.reshape(temp2, (1,-1))),axis=0)
				
				'''Middle Area'''
				temp3, temp4 = calProjection(motion_, photon_, [currPatternX[1],currPatternY[1]], center = left, focus = True)
				outer_mid = np.concatenate((outer_mid, np.reshape(temp3, (1,-1))),axis=0)
				inner_mid = np.concatenate((inner_mid, np.reshape(temp4, (1,-1))),axis=0)
				
				'''Outer Area'''
				temp5, temp6 = calProjection(motion_, photon_, [currPatternX[2],currPatternY[2]], center = left, focus = True)
				outer_out = np.concatenate((outer_out, np.reshape(temp5, (1,-1))),axis=0)
				inner_out = np.concatenate((inner_out, np.reshape(temp6, (1,-1))),axis=0)
			
			temp_rt = np.column_stack([euler_pred_onedata,
                              # hand_pred_onedata,
                              gyro_pred_onedata,
                              # hand_gyro_pred_onedata,
                              alfa_pred_onedata,
                              # position_pred_onedata
                              ])
#			temp_rt = tempNorm.transform(temp_rt)
			x_seq_rt[:-1] = x_seq_rt[1:]
			x_seq_rt[-1] = temp_rt

			if rt_counter < DELAY_SIZE + anticipation_size:
				rt_counter +=1
            
#			seq_df = pd.DataFrame(x_seq_rt)
#			seq_df = seq_df.rolling(sliding_window_size, min_periods=1).mean()
#			temp_seq_rt = np.array(seq_df)
#			xd_seq_rt[:-1] = xd_seq_rt[1:]
#			xd_seq_rt[-1] = temp_seq_rt[-1]

#
#		if graphShow == 'movement':
#			if coordinate == 'pitch':
#				y_plot[0] = train_eule_data[nowTime][0]
#				#CRP
#				y_plot[1] = tempCrp[0][0]
#				#CAP
#				y_plot[2] = tempCap[0][0]
#				#NOP
#				y_plot[3] = tempNop[0][0]
#				#ANN
#				y_plot[4] = y_sample[0][0]
#			elif coordinate == 'roll':
#				y_plot[0] = train_eule_data[nowTime][1]
#				#CRP
#				y_plot[1] = tempCrp[0][1]
#				#CAP
#				y_plot[2] = tempCap[0][1]
#				#NOP
#				y_plot[3] = tempNop[0][1]
#				#ANN
#				y_plot[4] = y_sample[0][1]
#			elif coordinate == 'yaw':
#				y_plot[0] = train_eule_data[nowTime][2]
#				#CRP
#				y_plot[1] = tempCrp[0][2]
#				#CAP
#				y_plot[2] = tempCap[0][2]
#				#NOP
#				y_plot[3] = tempNop[0][2]
#				#ANN
#				y_plot[4] = y_sample[0][2]
#		rtp.RealTimePlot(float(timestamp_plot_onedata), y_plot)

# #####################################With SG Filter
#		# Gather data until minimum delay is fulfilled
#		if rt_counter < DELAY_SIZE:
#			temp_rt = np.column_stack([euler_pred_onedata, gyro_pred_onedata, alfa_pred_onedata])
#			temp_rt = tempNorm.transform(temp_rt)
#			x_seq_rt[:-1] = x_seq_rt[1:]
#			x_seq_rt[-1] = temp_rt
#			rt_counter += 1
#			continue			
#		else:
#			y_NN = new_sess.run(y, feed_dict={x:x_seq_rt.reshape(1,DELAY_SIZE,input_nm)})
#			ann_dummy_rt = np.concatenate((ann_dummy_rt, y_NN), axis =0)
#			ann_pred_rt = np.concatenate((ann_pred_rt, y_NN), axis =0)
#			
#			#Get temp cap crp and nop
#			tempCap = cap_prediction(euler_pred_onedata, gyro_pred_onedata* np.pi / 180, alfa_pred_onedata, anticipation_time)
#			tempCrp = crp_prediction(euler_pred_onedata, gyro_pred_onedata* np.pi / 180, anticipation_time)
#			tempNop = nop_prediction(euler_pred_onedata, anticipation_time)
#			
#			cap_pred_rt = np.concatenate((cap_pred_rt, tempCap), axis = 0)
#			crp_pred_rt = np.concatenate((crp_pred_rt, tempCrp), axis = 0)
#			nop_pred_rt = np.concatenate((nop_pred_rt, tempNop), axis = 0)
#			
#			temp_rt = np.column_stack([euler_pred_onedata, gyro_pred_onedata, alfa_pred_onedata])
#			temp_rt = tempNorm.transform(temp_rt)
#			x_seq_rt[:-1] = x_seq_rt[1:]
#			x_seq_rt[-1] = temp_rt
#	
#			if (m == 0):
#				#Array less than ideal input(Sg sliding windows and end buff sliding)
#				if (k<SG_window_size+buff_window):
#					k = k+1
#					continue
#				else:
#					#Perform savgol only without optimal buffer
#					temp = signal.savgol_filter(ann_dummy_rt[0:k,:], SG_window_size, 3, axis=0)
#					ann_pred_rt[0:SG_window_size,:] = temp[0:SG_window_size,:]
#					m = m+1
#					k = 0
#			elif(i<(len(euler_o)-anticipation_size-1)):
#				#Array less than windows size(sg sliding windows)
#				if (k<SG_window_size):
#					k = k+1
#				else:
#					#If necesarry input is fulfilled
#					temp = signal.savgol_filter(ann_dummy_rt[m*SG_window_size-buff_window:m*SG_window_size + k + buff_window,:], SG_window_size, 3, axis=0)
#					ann_pred_rt[m*SG_window_size:m*SG_window_size+k,:] = temp[buff_window:-buff_window,:]
#					m = m+1
#					k = 0
#			else:
#				makst = math.floor((len(euler_o)-anticipation_size)/SG_window_size)
#				k = SG_window_size
#				for t in range(m,makst):
#					if (t < makst-1):
#						temp = signal.savgol_filter(ann_dummy_rt[t*SG_window_size-buff_window:t*SG_window_size + k + buff_window,:], SG_window_size, 3, axis=0)
#						ann_pred_rt[t*SG_window_size:t*SG_window_size+k,:] = temp[buff_window:-buff_window,:]
#					else:
#						temp = signal.savgol_filter(ann_dummy_rt[t*SG_window_size-buff_window::,:], SG_window_size, 3, axis=0)
#						ann_pred_rt[t*SG_window_size::,:] = temp[buff_window::,:]
# 				
## 		#Plotting realtime
#		y_sample = ann_pred_rt[i-(SG_window_size + buff_window)].reshape(1,-1)
#		tempCap = cap_pred_rt[i-(SG_window_size + buff_window)].reshape(1,-1)
#		tempCrp = crp_pred_rt[i-(SG_window_size + buff_window)].reshape(1,-1)
#		tempNop = nop_pred_rt[i-(SG_window_size + buff_window)].reshape(1,-1)
#		tempActual = train_eule_data[nowTime-(SG_window_size + buff_window)].reshape(1,-1)
#		
#		if graphShow == 'movement':
#			if coordinate == 'pitch':
#				y_plot[0] = tempActual[0][0]
#				#CRP
#				y_plot[1] = tempCrp[0][0]
#				#CAP
#				y_plot[2] = tempCap[0][0]
#				#NOP
#				y_plot[3] = tempNop[0][0]
#				#ANN
#				y_plot[4] = y_sample[0][0]
#			elif coordinate == 'roll':
#				y_plot[0] = tempActual[0][1]
#				#CRP
#				y_plot[1] = tempCrp[0][1]
#				#CAP
#				y_plot[2] = tempCap[0][1]
#				#NOP
#				y_plot[3] = tempNop[0][1]
#				#ANN
#				y_plot[4] = y_sample[0][1]
#			elif coordinate == 'yaw':
#				y_plot[0] = tempActual[0][2]
#				#CRP
#				y_plot[1] = tempCrp[0][2]
#				#CAP
#				y_plot[2] = tempCap[0][2]
#				#NOP
#				y_plot[3] = tempNop[0][2]
#				#ANN
#				y_plot[4] = y_sample[0][2]
#		rtp.RealTimePlot(float(timestamp_plot_onedata), y_plot)

for i in range (anticipation_size):
	overfilling_rt = np.concatenate((overfilling_rt, np.reshape(proj_data[0], (1, -1))), axis = 0)
# 	ido_values = np.vstack((ido_values,proj_data[0]))
# 	ellips_values = np.vstack((ellips_values,proj_data[0]))
	gaze_pos = np.vstack((gaze_pos,[(proj_data[i,0]+proj_data[i,2])/2,(proj_data[i,3]+proj_data[i,1])/2]))
	radiiPattern = np.vstack((radiiPattern,[0.25,0.35,10, 0.25,0.35,10]))

"""Calculate online error"""
#Error is defined as difference between predicted head orientation. 
#Actual head orientation = Current head orientation shifted by s time
euler_ann_err_rt = np.abs(ann_pred_rt[:,0:3] - train_eule_data[anticipation_size+DELAY_SIZE:-(anticipation_size)])
euler_cap_err_rt = np.abs(cap_pred_rt - train_eule_data[anticipation_size+DELAY_SIZE:-(anticipation_size)])
euler_crp_err_rt = np.abs(crp_pred_rt - train_eule_data[anticipation_size+DELAY_SIZE:-(anticipation_size)])
euler_nop_err_rt = np.abs(nop_pred_rt - train_eule_data[anticipation_size+DELAY_SIZE:-(anticipation_size)])
# hand_euler_ann_err_rt = np.abs(ann_pred_rt[:,3:6] - hand_euler_o[anticipation_size:])
# position_ann_err_rt = np.abs(ann_pred_rt[:,6::] - position_o[anticipation_size:])

patternX = outer[:,1]
patternY = outer[:,2]
HighAmount = np.pi*patternX*patternY/((proj_data[0,2]-proj_data[0,0])*(proj_data[0,1]-proj_data[0,3]))*100

patternX_mid = outer_mid[:,1]
patternY_mid = outer_mid[:,2]
MidAmount = (np.pi*patternX_mid*patternY_mid- np.pi*patternX*patternY)/((proj_data[0,2]-proj_data[0,0])*(proj_data[0,1]-proj_data[0,3]))*100

OutAmount =  100 - HighAmount - MidAmount

proj_size = (overfilling_rt[:,2]-overfilling_rt[:,0])*(overfilling_rt[:,1]-overfilling_rt[:,3])
overfill_amount = (proj_size/((proj_data[0,2]-proj_data[0,0])*(proj_data[0,1]-proj_data[0,3]))-1)*100

# proj_size = (pred_proj_data[anticipation_size+DELAY_SIZE:-anticipation_size,2]-pred_proj_data[anticipation_size+DELAY_SIZE:-anticipation_size,0])*(pred_proj_data[anticipation_size+DELAY_SIZE:-anticipation_size,1]-pred_proj_data[anticipation_size+DELAY_SIZE:-anticipation_size,3])
# overfill_actual = (proj_size/((proj_data[0,2]-proj_data[0,0])*(proj_data[0,1]-proj_data[0,3]))-1)*100

# proj_size = (ido_values[:,2]-ido_values[:,0])*(ido_values[:,1]-ido_values[:,3])
# oversize_ido = (proj_size/((proj_data[0,2]-proj_data[0,0])*(proj_data[0,1]-proj_data[0,3]))-1)*100

# proj_size = (ellips_values[:,2]-ellips_values[:,0])*(ellips_values[:,1]-ellips_values[:,3])
# oversize_ellips = (proj_size/((proj_data[0,2]-proj_data[0,0])*(proj_data[0,1]-proj_data[0,3]))-1)*100


print('Overfill Average = {:.2f}'.format(np.nanmean(overfill_amount)))

"""Write to CSV"""
raw_robust_data = np.column_stack(
	[timestamp_plot*705600000, 
	position_o[anticipation_size:,0:6], 
	eul2quat_bio(euler_o)[anticipation_size:,[3,1,2,0]],
	euler_o[anticipation_size:,[2,0,1]],
	accel_o[anticipation_size:,0:3],
	gyro_o[anticipation_size:], 
	proj_data[anticipation_size+DELAY_SIZE:-anticipation_size],
	position_o[anticipation_size:,6:9],
	eul2quat_bio(hand_euler_o)[anticipation_size:],
	hand_euler_o[anticipation_size:],
	accel_o[anticipation_size:,3:6],
	train_hand_gyro_data[anticipation_size+DELAY_SIZE:-anticipation_size],
	pred_time[anticipation_size+DELAY_SIZE:-anticipation_size],
    position_o[anticipation_size:,0:6],
# 	eul2quat_bio(ann_pred_rt[:,0:3]),
# 	ann_pred_rt[:,0:3],
	eul2quat_bio(euler_o)[anticipation_size:,[3,1,2,0]],
	euler_o[anticipation_size:,[2,0,1]],
 	overfilling_rt[anticipation_size:],
 	# ellips_values,
# 	ido_values,
    radiiPattern[anticipation_size:,0],
    radiiPattern[anticipation_size:,1],
	position_o[anticipation_size:,6:9],
	eul2quat_bio(hand_euler_o)[anticipation_size:],
	hand_euler_o[anticipation_size:],
	]
)

# write the predicted into file 
df = pd.DataFrame(raw_robust_data,columns = ['timestamp',
                            'input_left_eye_position_x', 'input_left_eye_position_y', 'input_left_eye_position_z',
                            'input_right_eye_position_x', 'input_right_eye_position_y',	'input_right_eye_position_z',
                            'input_head_orientation_x', 'input_head_orientation_y', 'input_head_orientation_z', 'input_head_orientation_w',
                            'input_head_orientation_yaw', 'input_head_orientation_pitch', 'input_head_orientation_roll',
                            'input_head_acceleration_x', 'input_head_acceleration_y', 'input_head_acceleration_z',
                            'input_head_angular_vec_x',	'input_head_angular_vec_y', 'input_head_angular_vec_z',
                            'input_camera_projection_left', 'input_camera_projection_top', 'input_camera_projection_right', 'input_camera_projection_bottom',
                            'input_right_hand_position_x' ,'input_right_hand_position_y', 'input_right_hand_position_z',
                            'input_right_hand_orientation_x', 'input_right_hand_orientation_y', 'input_right_hand_orientation_z', 'input_right_hand_orientation_w',
                            'input_right_hand_orientation_yaw',	'input_right_hand_orientation_pitch', 'input_right_hand_orientation_roll',
                            'input_right_hand_acceleration_x', 'input_right_hand_acceleration_y', 'input_right_hand_acceleration_z',
                            'input_right_hand_angular_vec_x', 'input_right_hand_angular_vec_y', 'input_right_hand_angular_vec_z',
							'prediction_time',
                            'predicted_left_eye_position_x', 'predicted_left_eye_position_y', 'predicted_left_eye_position_z',
                            'predicted_right_eye_position_x', 'predicted_right_eye_position_y', 'predicted_right_eye_position_z',
							'predicted_head_orientation_x', 'predicted_head_orientation_y', 'predicted_head_orientation_z', 'predicted_head_orientation_w',
							'predicted_head_orientation_yaw', 'predicted_head_orientation_pitch', 'predicted_head_orientation_roll',
							'predicted_camera_projection_left', 'predicted_camera_projection_top', 'predicted_camera_projection_right', 'predicted_camera_projection_bottom',
							'predicted_foveation_inner_radius', 'predicted_foveation_middle_radius',
                            'predicted_right_hand_position_x', 'predicted_right_hand_position_y', 'predicted_right_hand_position_z',
							'predicted_right_hand_orientation_x', 'predicted_right_hand_orientation_y', 'predicted_right_hand_orientation_z', 'predicted_right_hand_orientation_w',
							'predicted_right_hand_orientation_yaw',	'predicted_right_hand_orientation_pitch', 'predicted_right_hand_orientation_roll',
                            ])
export_csv = df.to_csv (str(infile)+"_output.csv", index = None, header=True)

'''
timestamp	input_left_eye_position_x	input_left_eye_position_y	input_left_eye_position_z
input_right_eye_position_x	input_right_eye_position_y	input_right_eye_position_z	
input_head_orientation_x	input_head_orientation_y	input_head_orientation_z	input_head_orientation_w	
input_head_orientation_yaw	input_head_orientation_pitch	input_head_orientation_roll	
input_head_acceleration_x	input_head_acceleration_y	input_head_acceleration_z	
input_head_angular_vec_x	input_head_angular_vec_y	input_head_angular_vec_z	
input_camera_projection_left	input_camera_projection_top	input_camera_projection_right	input_camera_projection_bottom	
input_right_hand_position_x	input_right_hand_position_y	input_right_hand_position_z	
input_right_hand_orientation_x	input_right_hand_orientation_y	input_right_hand_orientation_z	input_right_hand_orientation_w	
input_right_hand_orientation_yaw	input_right_hand_orientation_pitch	input_right_hand_orientation_roll	
input_right_hand_acceleration_x	input_right_hand_acceleration_y	input_right_hand_acceleration_z	
input_right_hand_angular_vec_x	input_right_hand_angular_vec_y	input_right_hand_angular_vec_z	
prediction_time	
predicted_left_eye_position_x	predicted_left_eye_position_y	predicted_left_eye_position_z	
predicted_right_eye_position_x	predicted_right_eye_position_y	predicted_right_eye_position_z	
predicted_head_orientation_x	predicted_head_orientation_y	predicted_head_orientation_z	predicted_head_orientation_w	
predicted_head_orientation_yaw	predicted_head_orientation_pitch	predicted_head_orientation_roll	
predicted_camera_projection_left	predicted_camera_projection_top	predicted_camera_projection_right	predicted_camera_projection_bottom	
predicted_foveation_inner_radius	predicted_foveation_middle_radius
predicted_right_hand_position_x	predicted_right_hand_position_y	predicted_right_hand_position_z	
predicted_right_hand_orientation_x	predicted_right_hand_orientation_y	predicted_right_hand_orientation_z	predicted_right_hand_orientation_w	
predicted_right_hand_orientation_yaw	predicted_right_hand_orientation_pitch	predicted_right_hand_orientation_roll
'''


# Calculate average error
"""offline"""

ann_mae = np.nanmean(np.abs(euler_ann_err), axis=0)
cap_mae = np.nanmean(np.abs(euler_cap_err), axis=0)
crp_mae = np.nanmean(np.abs(euler_crp_err), axis=0)
nop_mae = np.nanmean(np.abs(euler_nop_err), axis=0)
# hand_ann_mae = np.nanmean(np.abs(hand_euler_ann_err), axis=0)
# position_ann_mae = np.nanmean(np.abs(position_ann_err), axis=0)

# Calculate max error
ann_max = np.nanmax(np.abs(euler_ann_err), axis=0)
cap_max = np.nanmax(np.abs(euler_cap_err), axis=0)
crp_max = np.nanmax(np.abs(euler_crp_err), axis=0)
nop_max = np.nanmax(np.abs(euler_nop_err), axis=0)

# Calculate 99% Percentile
final_ann_99 = np.nanpercentile(euler_ann_err,99, axis = 0)
final_cap_99 = np.nanpercentile(euler_cap_err,99, axis = 0)
final_crp_99 = np.nanpercentile(euler_crp_err,99, axis = 0)
final_nop_99 = np.nanpercentile(euler_nop_err,99, axis = 0)
# final_hand_ann_99 = np.nanpercentile(hand_euler_ann_err,99, axis = 0)
# final_position_ann_99 = np.nanpercentile(position_ann_err,99, axis = 0)


# get rms stream
ann_rms_stream = np.apply_along_axis(rms,1,euler_ann_err)
cap_rms_stream = np.apply_along_axis(rms,1,euler_cap_err)
crp_rms_stream = np.apply_along_axis(rms,1,euler_crp_err)
nop_rms_stream = np.apply_along_axis(rms,1,euler_nop_err)


# calculate error rms mean
ann_rms = np.nanmean(ann_rms_stream)
cap_rms = np.nanmean(cap_rms_stream)
crp_rms = np.nanmean(crp_rms_stream)
nop_rms = np.nanmean(nop_rms_stream)


"""online"""
ann_mae_rt = np.nanmean(np.abs(euler_ann_err_rt), axis=0)
cap_mae_rt = np.nanmean(np.abs(euler_cap_err_rt), axis=0)
crp_mae_rt = np.nanmean(np.abs(euler_crp_err_rt), axis=0)
nop_mae_rt = np.nanmean(np.abs(euler_nop_err_rt), axis=0)
# hand_ann_mae_rt = np.nanmean(np.abs(hand_euler_ann_err_rt), axis=0) 
# position_ann_mae_rt = np.nanmean(np.abs(position_ann_err_rt), axis=0)

#Max
ann_max_rt = np.nanmax(np.abs(euler_ann_err_rt), axis=0)
cap_max_rt = np.nanmax(np.abs(euler_cap_err_rt), axis=0)
crp_max_rt = np.nanmax(np.abs(euler_crp_err_rt), axis=0)
nop_max_rt = np.nanmax(np.abs(euler_nop_err_rt), axis=0)

#99MAE
final_ann_rt_99 = np.nanpercentile(euler_ann_err_rt,99, axis = 0)
final_cap_rt_99 = np.nanpercentile(euler_cap_err_rt,99, axis = 0)
final_crp_rt_99 = np.nanpercentile(euler_crp_err_rt,99, axis = 0)
final_nop_rt_99 = np.nanpercentile(euler_nop_err_rt,99, axis = 0)
# final_hand_ann_rt_99 = np.nanpercentile(hand_euler_ann_err_rt,99, axis = 0)
# final_position_ann_rt_99 = np.nanpercentile(position_ann_err_rt,99, axis = 0)

ann_rms_stream_rt = np.apply_along_axis(rms,1,euler_ann_err_rt)
cap_rms_stream_rt = np.apply_along_axis(rms,1,euler_cap_err_rt)
crp_rms_stream_rt = np.apply_along_axis(rms,1,euler_crp_err_rt)
nop_rms_stream_rt = np.apply_along_axis(rms,1,euler_nop_err_rt)
#
print('Offline - MAE')
print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann_mae[0], ann_mae[1], ann_mae[2]))
# print('Hand ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(hand_ann_mae[0], hand_ann_mae[1], hand_ann_mae[2]))
# print('R_eye Position ANN [x, y, z]: {:.2f}, {:.2f}, {:.2f}'.format(position_ann_mae[0], position_ann_mae[1], position_ann_mae[2]))
print('CAP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(cap_mae[0], cap_mae[1], cap_mae[2]))
print('CRP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(crp_mae[0], crp_mae[1], crp_mae[2]))
print('NOP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(nop_mae[0], nop_mae[1], nop_mae[2]))

print('Online - MAE')
print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(ann_mae_rt[0], ann_mae_rt[1], ann_mae_rt[2]))
# print('Hand ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(hand_ann_mae_rt[0], hand_ann_mae_rt[1], hand_ann_mae_rt[2]))
# print('R_eye Position ANN [x, y, z]: {:.2f}, {:.2f}, {:.2f}'.format(position_ann_mae_rt[0], position_ann_mae_rt[1], position_ann_mae_rt[2]))
print('CAP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(cap_mae_rt[0], cap_mae_rt[1], cap_mae_rt[2]))
print('CRP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(crp_mae_rt[0], crp_mae_rt[1], crp_mae_rt[2]))
print('NOP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(nop_mae_rt[0], nop_mae_rt[1], nop_mae_rt[2]))

print('\nOffline - 99% MAE')
print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_ann_99[0], final_ann_99[1], final_ann_99[2]))
# print('Hand ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_hand_ann_99[0], final_hand_ann_99[1], final_hand_ann_99[2]))
# print('R_eye ANN [x, y, z]: {:.2f}, {:.2f}, {:.2f}'.format(final_position_ann_99[0], final_position_ann_99[1], final_position_ann_99[2]))
print('CAP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(final_cap_99[0], final_cap_99[1], final_cap_99[2]))
print('CRP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(final_crp_99[0], final_crp_99[1], final_crp_99[2]))
print('NOP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(final_nop_99[0], final_nop_99[1], final_nop_99[2]))

print('Online - 99% MAE')
print('ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_ann_rt_99[0], final_ann_rt_99[1], final_ann_rt_99[2]))
# print('Hand ANN [Pitch, Roll, Yaw]: {:.2f}, {:.2f}, {:.2f}'.format(final_hand_ann_rt_99[0], final_hand_ann_rt_99[1], final_hand_ann_rt_99[2]))
# print('R_eye ANN [x, y, z]: {:.2f}, {:.2f}, {:.2f}'.format(final_position_ann_rt_99[0], final_position_ann_rt_99[1], final_position_ann_rt_99[2]))
print('CAP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(final_cap_rt_99[0], final_cap_rt_99[1], final_cap_rt_99[2]))
print('CRP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(final_crp_rt_99[0], final_crp_rt_99[1], final_crp_rt_99[2]))
print('NOP [Pitch, Roll, Yaw]: {:.2f},{:.2f},{:.2f}'.format(final_nop_rt_99[0], final_nop_rt_99[1], final_nop_rt_99[2]))

# plt.figure()
# plt.scatter(gaze_pos[:,0], gaze_pos[:,1], alpha = 0.5, s = 24)
# plt.scatter(gaze_pos[:,2], gaze_pos[:,3], alpha = 0.5, s = 24)
# plt.legend(['Right Eye', 'Left Eye'])
# plt.title('Gaze-Tracking Position on Quest')
# # plt.grid()
# plt.xlabel('X-Position (unit Scaled)')
# plt.ylabel('Y-Position (unit Scaled)')
# plt.show(block=False)

# plt.figure()
# plt.scatter(gaze_pos[:,0], gaze_pos[:,1], alpha = 0.5, s = 24)
# plt.scatter(gaze_pos_fixed[:,0], gaze_pos_fixed[:,1], alpha = 0.5, s = 24)
# plt.legend(['Dynamic', 'Fixed'])
# plt.title('Gaze-Tracking Right Eye Position on Quest')
# # plt.grid()
# plt.xlabel('X-Position (unit Scaled)')
# plt.ylabel('Y-Position (unit Scaled)')
# plt.show(block=False)

# plt.figure()
# plt.scatter(gaze_pos[:,2], gaze_pos[:,3], alpha = 0.5, s = 24)
# plt.scatter(gaze_pos_fixed[:,2], gaze_pos_fixed[:,3], alpha = 0.5, s = 24)
# plt.legend(['Dynamic', 'Fixed'])
# plt.title('Gaze-Tracking left Eye Position on Quest')
# # plt.grid()
# plt.xlabel('X-Position (unit Scaled)')
# plt.ylabel('Y-Position (unit Scaled)')
# plt.show(block=False)

# proj_size = (overfilling_rt[:,2]-overfilling_rt[:,0])*(overfilling_rt[:,1]-overfilling_rt[:,3])

# plt.figure()
# plt.plot(timestamp_plot,proj_size[:-anticipation_size])
# plt.plot(timestamp_plot, lowRes)
# plt.plot(timestamp_plot, middleRes)
# plt.plot(timestamp_plot, highRes)
# plt.legend(['Original', 'Peripheral', 'Middle', 'Inner'])
# plt.title('Shading Region on Multiple Pattern Level')
# # plt.grid()
# plt.xlabel('time-stamp (second)')
# plt.ylabel('Region Amount (unit Scaled)')
# plt.show(block=False)

# plt.figure()
# plt.plot(timestamp_plot, middleRes, color = "green")
# plt.plot(timestamp_plot, highRes, color = "red")
# plt.legend(['Middle', 'Inner'])
# plt.title('Shading Region on Multiple Pattern Level')
# # plt.grid()
# plt.xlabel('time-stamp (second)')
# plt.ylabel('Region Amount (unit Scaled)')
# plt.show(block=False)

# print('\nOriginal : {:.2f}'.format(np.nanmean(proj_size[:-anticipation_size])))
# print('Peripheral : {:.2f}'.format(np.nanmean(lowRes)))
# print('Middle : {:.2f}'.format(np.nanmean(middleRes)))
# print('Inner : {:.2f}'.format(np.nanmean(highRes)))


# plt.figure()
# x = np.sort((highRes/center_size-1)*100)
# y = np.arange(1, len(x)+1)/len(x)
# plt.plot(x, y, marker='.', linestyle='none')
# plt.xlabel('percentage of high resolution oversized center region (%)')
# plt.ylabel('likehood of Occurance')
# plt.title('CDF of percentage of high resolution oversized center region')
# plt.margins(0.02)

# m2DX, m2DY = createPixel(proj_data[0])

# '''Foveation Metrices on HighRes'''
# center_size = np.zeros(0, dtype= float)
# oversized = np.zeros(0, dtype= float)
# inn = np.zeros(0, dtype= float)
# mid = np.zeros(0, dtype= float)
# out = np.zeros(0, dtype= float)
# freq = np.zeros(0, dtype= float)

# length = 1000

# for i in range (length):
# 	if (i%200 == 0):
# 		print('Calculation Number: {:d}'.format(i))
# 	size1, size2, size3, size4, size5, res = calNumPixel(outer[i], outer_mid[i], outer_out[i], inner[i], proj_data[0], m2DX, m2DY, const = 0.1)
# 	
# 	inn = np.concatenate((inn, [size1]), axis = 0)
# 	oversized = np.concatenate((oversized, [size2]), axis = 0)
# 	center_size = np.concatenate((center_size, [size3]), axis = 0)
# 	mid = np.concatenate((mid, [size4]), axis = 0)
# 	out = np.concatenate((out, [size5]), axis = 0)
# 	
# 	freq = np.concatenate((freq,[res]), axis = 0)


# plt.figure()
# plt.plot(timestamp_plot[:len(inn)], inn, color = "blue")
# plt.plot(timestamp_plot[:len(oversized)], oversized, color = "green")
# plt.plot(timestamp_plot[:len(center_size)], center_size, color = "red")
# plt.plot(timestamp_plot[:len(center_size)], np.full(len(center_size), 100),'--', color = "black")
# plt.legend(['Ellips on Total User Screen', 'Oversized High-Res on Center-Rect', 'High-Res on Center-Rect', '100% Center-Rect size'])
# plt.title('Center-Rectangular vs High-Resolution Region')
# # plt.grid()
# plt.xlabel('time-stamp (second)')
# plt.ylabel('Percentage (%)')
# plt.show(block=False)

# plt.figure()
# plt.plot(timestamp_plot[:len(inn)], inn, color = "blue")
# plt.plot(timestamp_plot[:len(mid)], mid, color = "green")
# plt.plot(timestamp_plot[:len(out)], out, color = "red")
# plt.plot(timestamp_plot[:len(center_size)], np.full(len(center_size), 100),'--', color = "black")
# plt.legend(['Inner', 'Middle', 'Outer', 'Full Size'])
# plt.title('Foveation Level Region Comparison')
# # plt.grid()
# plt.xlabel('time-stamp (second)')
# plt.ylabel('Percentage (%)')
# plt.show(block=False)

# plt.figure()
# x = np.sort(center_size)
# y = np.arange(1, len(x)+1)/len(x)
# plt.plot(x, y, marker='.', linestyle='none')
# plt.xlabel('Percentage of User Screen covered by High Resolution (%)')
# plt.ylabel('likehood of Occurance')
# plt.title('CDF of Center-Rect covered by High Resolution')
# plt.margins(0.02)

# plt.show()

# print('\nPercentage of HighRes on Center-Rect: {:.2f}%'.format(np.nanmean(center_size)))

# plt.figure()
# x = np.sort(oversized)
# y = np.arange(1, len(x)+1)/len(x)
# plt.plot(x, y, marker='.', linestyle='none')
# plt.xlabel('Percentage of High Resolution oversized Center Size (%)')
# plt.ylabel('likehood of Occurance')
# plt.title('CDF of High Resolution oversized Center-Rect')
# plt.margins(0.02)

# plt.show()

# print('Percentage of oversized HighRes on Center-Rect Size: {:.2f}%'.format(np.nanmean(oversized)))

# '''Foveation Matrices on Contrast Sensitivity'''
# # length = 1000
# # result = np.zeros(0, dtype= float)
# # for i in range (length):
# # 	if (i%200 == 0):
# # 		print('Calculation Number: {:d}'.format(i))
# # 	result = np.concatenate((result,[contrastSensitivity(m2DX, m2DY, outer[i], inner[i], v = 1, CT = 1)]), axis = 0)

# '''Plot spatial with center Rect Freq'''
# # cPoint = [(proj_data[0,2]+proj_data[0,0])/2, (proj_data[0,1]+proj_data[0,3])/2]
# # f = contrastSensitivity(m2DX, m2DY, outer[0], inner[0], cPoint = cPoint,  v = 1, CT = 1)
# # side = np.sqrt((proj_data[0,2]-proj_data[0,0])*(proj_data[0,1]-proj_data[0,3])*0.1)
# # side2 = np.sqrt((proj_data[0,2]-proj_data[0,0])*(proj_data[0,1]-proj_data[0,3])*0.08)
# # side3 = np.sqrt((proj_data[0,2]-proj_data[0,0])*(proj_data[0,1]-proj_data[0,3])*0.05)

# # plt.figure()
# # plt.plot(np.full(100,-side/2+cPoint[0]), np.linspace(0,35,100), color = "orange")
# # plt.plot(np.full(100,-side2/2+cPoint[0]), np.linspace(0,35,100), color = "green")
# # plt.plot(np.full(100,-side3/2+cPoint[0]), np.linspace(0,35,100), color = "red")
# # plt.plot(np.full(100,+side/2+cPoint[0]), np.linspace(0,35,100), color = "orange")
# # plt.plot(np.full(100,+side2/2+cPoint[0]), np.linspace(0,35,100), color = "green")
# # plt.plot(np.full(100,+side3/2+cPoint[0]), np.linspace(0,35,100), color = "red")
# # # plt.plot(m2DX,f[:,np.where(m2DY == min(abs(m2DY)))[0]], marker='.', linestyle='none') #np.linspace(-512,512,1024)
# # plt.plot(m2DX, f[:,512], marker='.', linestyle='none') #np.linspace(-512,512,1024)
# # plt.xlabel('Image Scale (unit)')
# # plt.ylabel('Spatial Frequency (cycles/degree)')
# # plt.title('Contrast Sensitivity')
# # plt.legend(['10% center size', ' 8% center size', '5% center size'])
# # plt.margins(0.02)

# plt.figure()
# x = np.sort(freq)
# y = np.arange(1, len(x)+1)/len(x)
# plt.plot(x, y, marker='.', linestyle='none')
# plt.xlabel('Spatial Frequency (cycle/degree)')
# plt.ylabel('likehood of Occurance')
# plt.title('CDF of Spatial Frequency Score(cycles/degree) of ' + scheme)
# plt.legend(['99%-tile Frequency: {:.2f} cycles/degree'.format(np.nanpercentile(freq,1))])
# plt.margins(0.02)

# plt.show()

# print('\nAverage Frequency of '+scheme+': {:.2f} cycles/degree'.format(np.nanmean(freq)))

# fov_center = center_size
# fov_over = oversized
# fov_inn = inn
# fov_mid = mid
# fov_out = out
# fov_freq = freq
