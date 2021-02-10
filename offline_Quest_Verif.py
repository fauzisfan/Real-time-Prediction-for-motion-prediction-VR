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
#from convention_biosignal import convention_biosignal
#from convention_biosignal_quat import convention_biosignal_quat   
import os
import math
# from sklearn.externals import joblib
import joblib

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

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
    input_angley = np.arctan(IPDx/(2*h))
    	
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
    
# In[5]:

anticipation_time = 300
'''
#####   SYSTEM INITIALIZATION    #####
'''
tf.reset_default_graph()

tf.set_random_seed(2)

np.random.seed(2)


#parser = argparse.ArgumentParser(description='Offline Motion Prediction')
#parser.add_argument('-a', '--anticipation', default=300, type=int)

#args = parser.parse_args()
infile = "Quest_20210201_scene(3)_user(1)"

try:
    stored_df = pd.read_csv(str(infile)+'_cut.csv')
    train_gyro_data = np.array(stored_df[['input_head_angular_vec_x', 'input_head_angular_vec_y', 'input_head_angular_vec_z']], dtype=np.float)
    train_hand_gyro_data = np.array(stored_df[['input_right_hand_angular_vec_x', 'input_right_hand_angular_vec_y', 'input_right_hand_angular_vec_z']], dtype=np.float)
    train_acce_data = np.array(stored_df[['input_head_acceleration_x', 'input_head_acceleration_y', 'input_head_acceleration_z','input_right_hand_acceleration_x', 'input_right_hand_acceleration_y', 'input_right_hand_acceleration_z']], dtype=np.float)
    train_eule_data = np.array(stored_df[['input_head_orientation_pitch', 'input_head_orientation_roll', 'input_head_orientation_yaw']], dtype=np.float)
    train_hand_eule_data = np.array(stored_df[['input_right_hand_orientation_pitch', 'input_right_hand_orientation_roll', 'input_right_hand_orientation_yaw']], dtype=np.float)
    train_position_data = np.array(stored_df[['input_right_eye_position_x','input_right_eye_position_y','input_right_eye_position_z', 'input_left_eye_position_x','input_left_eye_position_y','input_left_eye_position_z','input_right_hand_position_x','input_right_hand_position_y','input_right_hand_position_z']])
    proj_data = np.array(stored_df[['input_camera_projection_left', 'input_camera_projection_top', 'input_camera_projection_right', 'input_camera_projection_bottom']], dtype=np.float32)
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
TEST_SIZE = 0.5
TRAIN_SIZE = 1 - TEST_SIZE

print("DELAY_SIZE =", DELAY_SIZE)

# Variables
TRAINED_MODEL_NAME = './Quest_realtime_model'

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
#x_train, t_train, x_test, t_test = train_test_split_tdnn(x_seq, target_series, TEST_SIZE)

""""""""" Old Preprocessong """""""""
# Save it
scaler_file = "Quest_realtime_scaller.save"
#
## Load it 
tempNorm = joblib.load(scaler_file)
# normalizer = preprocessing.StandardScaler()

#Get normalized based data on the 
# tempNorm = normalizer.fit(input_series)
#Normalizer used on input series
input_norm = tempNorm.transform(input_series)


# Reformat the input into TDNN format
x_seq, t_seq = preparets(input_norm, target_series, DELAY_SIZE)
data_length = x_seq.shape[0]
print('Anticipation time: {}ms\n'.format(anticipation_time))


# Reset the whole tensorflow graph
tf.reset_default_graph()


# Split training and testing data
x_train, t_train, x_test, t_test = train_test_split_tdnn(x_seq, t_seq, TEST_SIZE)


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
        # layers.Dense(20, activation=tf.nn.relu, use_bias=True, kernel_regularizer=regularizers.l2(0.01)),
        # layers.Dropout(0.2),
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
gyro_o = train_gyro_data[DELAY_SIZE:-anticipation_size]* np.pi / 180
alfa_o = train_alfa_data[DELAY_SIZE:-anticipation_size]
accel_o = train_acce_data[DELAY_SIZE:-anticipation_size]
velocity_o = train_velocity_data[:-anticipation_size]

# Predict orientation
euler_pred_ann = y_out
euler_pred_cap = cap_prediction(euler_o, gyro_o, alfa_o, anticipation_time)
euler_pred_crp = crp_prediction(euler_o, gyro_o, anticipation_time)
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

# Plot
timestamp_plot = train_time_data[DELAY_SIZE:-2*anticipation_size]
time_offset = timestamp_plot[0]
timestamp_plot = np.array(timestamp_plot)-time_offset

plt.figure()
plt.plot(timestamp_plot, euler_pred_ann[:-anticipation_size, 2], linewidth=1,color='magenta')
#plt.plot(timestamp_plot, euler_pred_cap[:-anticipation_size, 2], linewidth=1,color='green')
#plt.plot(timestamp_plot, euler_pred_crp[:-anticipation_size, 2], linewidth=1,color='blue')
#plt.plot(timestamp_plot, euler_pred_nop[:-anticipation_size, 2], linewidth=1)
plt.plot(timestamp_plot, euler_o[anticipation_size:, 2], linewidth=1, color='navy')
plt.legend(['ANN','Actual'])
plt.title('Orientation Prediction (Yaw)')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Orientation (deg)')
plt.show(block=False)

plt.figure()
plt.plot(timestamp_plot, euler_pred_ann[:-anticipation_size, 1], linewidth=1,color='magenta')
#plt.plot(timestamp_plot, euler_pred_cap[:-anticipation_size, 2], linewidth=1,color='green')
#plt.plot(timestamp_plot, euler_pred_crp[:-anticipation_size, 2], linewidth=1,color='blue')
#plt.plot(timestamp_plot, euler_pred_nop[:-anticipation_size, 2], linewidth=1)
plt.plot(timestamp_plot, euler_o[anticipation_size:, 1], linewidth=1, color='navy')
plt.legend(['ANN','Actual'])
plt.title('Orientation Prediction (Roll)')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Orientation (deg)')
plt.show(block=False)

plt.figure()
plt.plot(timestamp_plot, euler_pred_ann[:-anticipation_size, 0], linewidth=1,color='magenta')
#plt.plot(timestamp_plot, euler_pred_cap[:-anticipation_size, 2], linewidth=1,color='green')
#plt.plot(timestamp_plot, euler_pred_crp[:-anticipation_size, 2], linewidth=1,color='blue')
#plt.plot(timestamp_plot, euler_pred_nop[:-anticipation_size, 2], linewidth=1)
plt.plot(timestamp_plot, euler_o[anticipation_size:, 0], linewidth=1, color='navy')
plt.legend(['ANN','Actual'])
plt.title('Orientation Prediction (Pitch)')
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Orientation (deg)')
plt.show(block=False)

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
error_rt = np.zeros((0,3))

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

with tf.Session() as new_sess:    
	model.load_weights(TRAINED_MODEL_NAME)
	for i in range(0,(len(train_eule_data)- (2*anticipation_size))):
		#Get euler, gyro, and alfa one by one
		nowTime = i + anticipation_size
		velocity_onedata = velocity_o[i].reshape(1,-1)
		
		euler_pred_onedata = solve_discontinuity(train_eule_data[i].reshape(1,-1))
# 		hand_pred_onedata = solve_discontinuity(train_hand_eule_data[i].reshape(1,-1))
		gyro_pred_onedata = train_gyro_data[i].reshape(1,-1)
		hand_gyro_pred_onedata = train_hand_gyro_data[i].reshape(1,-1)
		position_pred_onedata = train_position_data[i].reshape(1,-1)
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
			rt_counter += 1
#			if (rt_counter ==sliding_window_size):
#				seq_df = pd.DataFrame(x_seq_rt)
#				seq_df = seq_df.rolling(sliding_window_size, min_periods=1).mean()
#				temp_seq_rt = np.array(seq_df)
#				xd_seq_rt[:-1] = xd_seq_rt[1:]
#				xd_seq_rt[-1] = temp_seq_rt[-1]
#			else:
#				xd_seq_rt[:-1] = xd_seq_rt[1:]
#				xd_seq_rt[-1] = temp_rt
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

#             #SoftSwitching
# 			for j in range(0,3):
# 				xin = (np.abs(velocity_onedata[:,j])-midVel[j])/avgVel[j]
# 				alfa = sigmoid(xin)
# 				y_sample[:,j] = alfa*y_sample[:,j] + (1-alfa)*tempNop[:,j]

			margin = Robust_overfilling(euler_pred_onedata[0]*np.pi/180, y_sample[0]*np.pi/180, proj_data[0], offset = 1.1, fixed_param = 1.3)
            
			ann_pred_rt = np.concatenate((ann_pred_rt, y_sample), axis =0)
			overfilling_rt = np.concatenate((overfilling_rt, np.reshape(margin, (1, -1))), axis = 0)
			
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
    overfilling_rt = np.concatenate((overfilling_rt, np.reshape([-1,1,1,-1], (1, -1))), axis = 0)

"""Calculate online error"""
#Error is defined as difference between predicted head orientation. 
#Actual head orientation = Current head orientation shifted by s time
euler_ann_err_rt = np.abs(ann_pred_rt[:,0:3] - train_eule_data[anticipation_size+DELAY_SIZE:-(anticipation_size)])
euler_cap_err_rt = np.abs(cap_pred_rt - train_eule_data[anticipation_size+DELAY_SIZE:-(anticipation_size)])
euler_crp_err_rt = np.abs(crp_pred_rt - train_eule_data[anticipation_size+DELAY_SIZE:-(anticipation_size)])
euler_nop_err_rt = np.abs(nop_pred_rt - train_eule_data[anticipation_size+DELAY_SIZE:-(anticipation_size)])
# hand_euler_ann_err_rt = np.abs(ann_pred_rt[:,3:6] - hand_euler_o[anticipation_size:])
# position_ann_err_rt = np.abs(ann_pred_rt[:,6::] - position_o[anticipation_size:])

proj_size = (overfilling_rt[:,2]-overfilling_rt[:,0])*(overfilling_rt[:,1]-overfilling_rt[:,3])
overfill_amount = (proj_size/4-1)*100
print('Overfill Average = {:.2f}'.format(np.nanmean(overfill_amount)))

"""Write to CSV"""
raw_robust_data = np.column_stack(
	[timestamp_plot*705600000, 
	position_o[anticipation_size:,0:6], 
	eul2quat_bio(euler_o)[anticipation_size:],
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
	eul2quat_bio(ann_pred_rt[:,0:3]),# use quat_data for input
	ann_pred_rt[:,0:3],# use orie_data for input
	overfilling_rt[anticipation_size:],
    position_o[anticipation_size:,6:9],
    eul2quat_bio(hand_euler_o)[anticipation_size:],
    hand_euler_o[anticipation_size:],
	]
)

# write the predicted into file 
df = pd.DataFrame(raw_robust_data,columns = ['timestamp',
                            'input_left_eye_position_x', 'input_left_eye_position_y', 'input_left_eye_position_z',
                            'input_right_eye_position_x', 'input_right_eye_position_y',	'input_right_eye_position_z',
                            'input_head_orientation_w', 'input_head_orientation_y', 'input_head_orientation_z', 'input_head_orientation_x',
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
							'predicted_head_orientation_w', 'predicted_head_orientation_y', 'predicted_head_orientation_z', 'predicted_head_orientation_x',
							'predicted_head_orientation_yaw', 'predicted_head_orientation_pitch', 'predicted_head_orientation_roll',
							'predicted_camera_projection_left', 'predicted_camera_projection_top', 'predicted_camera_projection_right', 'predicted_camera_projection_bottom',
							'predicted_right_hand_position_x', 'predicted_right_hand_position_y', 'predicted_right_hand_position_z',
							'predicted_right_hand_orientation_x', 'predicted_right_hand_orientation_y', 'predicted_right_hand_orientation_z', 'predicted_right_hand_orientation_w',
							'predicted_right_hand_orientation_yaw',	'predicted_right_hand_orientation_pitch', 'predicted_right_hand_orientation_roll',
                            ])
export_csv = df.to_csv (str(infile)+"_output.csv", index = None, header=True)

# Calculate average error
"""offline"""

# Split error value
euler_ann_err_train = euler_ann_err[: int(TRAIN_SIZE*data_length)] 
euler_cap_err_train = euler_cap_err[: int(TRAIN_SIZE*data_length)]
euler_crp_err_train = euler_crp_err[: int(TRAIN_SIZE*data_length)]
euler_nop_err_train = euler_nop_err[: int(TRAIN_SIZE*data_length)]

euler_ann_err_test = euler_ann_err[int((1-TEST_SIZE)*data_length):] 
euler_cap_err_test = euler_cap_err[int((1-TEST_SIZE)*data_length):] 
euler_crp_err_test = euler_crp_err[int((1-TEST_SIZE)*data_length):] 
euler_nop_err_test = euler_nop_err[int((1-TEST_SIZE)*data_length):]

ann_mae = np.nanmean(np.abs(euler_ann_err), axis=0)
cap_mae = np.nanmean(np.abs(euler_cap_err), axis=0)
crp_mae = np.nanmean(np.abs(euler_crp_err), axis=0)
nop_mae = np.nanmean(np.abs(euler_nop_err), axis=0)
# hand_ann_mae = np.nanmean(np.abs(hand_euler_ann_err), axis=0)
# position_ann_mae = np.nanmean(np.abs(position_ann_err), axis=0)

ann_mae_train = np.nanmean(np.abs(euler_ann_err_train), axis=0)
cap_mae_train = np.nanmean(np.abs(euler_cap_err_train), axis=0)
crp_mae_train = np.nanmean(np.abs(euler_crp_err_train), axis=0)
nop_mae_train = np.nanmean(np.abs(euler_nop_err_train), axis=0)

ann_mae_test = np.nanmean(np.abs(euler_ann_err_test), axis=0)
cap_mae_test = np.nanmean(np.abs(euler_cap_err_test), axis=0)
crp_mae_test = np.nanmean(np.abs(euler_crp_err_test), axis=0)
nop_mae_test = np.nanmean(np.abs(euler_nop_err_test), axis=0)

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

final_ann_99_test = np.nanpercentile(euler_ann_err_test,99, axis = 0)
final_cap_99_test = np.nanpercentile(euler_cap_err_test,99, axis = 0)
final_crp_99_test = np.nanpercentile(euler_crp_err_test,99, axis = 0)
final_nop_99_test = np.nanpercentile(euler_nop_err_test,99, axis = 0)

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