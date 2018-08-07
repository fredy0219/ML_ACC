import numpy as np
from scipy import signal 
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import math

##############################################
## Get the stander of raw datas
##############################################
def get_stander(raw_x, raw_y, raw_z):

	return np.mean(raw_x), np.mean(raw_y), np.mean(raw_z)

##############################################
## Get the mean of raw datas
##############################################
def get_mean_data( raw_x, raw_y, raw_z , stander_x, stander_y, stander_z):

	return raw_x-stander_x, raw_y-stander_y, raw_z-stander_z

##############################################
## Get the filter of raw datas
##############################################
def get_filter_data( raw_x, raw_y, raw_z ):
	return signal.savgol_filter(raw_x, 15, 2), signal.savgol_filter(raw_y, 15, 2), signal.savgol_filter(raw_z, 15, 2)

def find_slope( raw_x, raw_y, raw_z ):

	min_delta_slope = 50

	pm2 = 0.25
	pm1 = 0.5
	p0 = 1.0
	pp1 = 0.5

	slope_list_x = []
	slope_list_y = []
	slope_list_z = []

	for i in range(len(raw_x)):
		if i > 2 and i < len(raw_x)-1:
			slope_list_x.append( abs(pm2 * abs(raw_x[i-2]) + pm1 * abs(raw_x[i-1]) + p0 * abs(raw_x[i]) + pp1 * abs(raw_x[i+1])))
			slope_list_y.append( abs(pm2 * abs(raw_y[i-2]) + pm1 * abs(raw_y[i-1]) + p0 * abs(raw_y[i]) + pp1 * abs(raw_y[i+1])))
			slope_list_z.append( abs(pm2 * abs(raw_z[i-2]) + pm1 * abs(raw_z[i-1]) + p0 * abs(raw_z[i]) + pp1 * abs(raw_z[i+1])))
		else:
			slope_list_x.append( 0 )
			slope_list_y.append( 0 )
			slope_list_z.append( 0 )


	# return np.array(slope_list) , signal_channel_segmentation(slope_list)
	return np.array(slope_list_x), np.array(slope_list_y), np.array(slope_list_z)

def segmentation(slope_x, slope_y, slope_z):

	segmentation_x = signal_channel_segmentation(slope_x)
	segmentation_y = signal_channel_segmentation(slope_y)
	segmentation_z = signal_channel_segmentation(slope_z)


	return sum_segmentation(segmentation_x,segmentation_y,segmentation_z) ,segmentation_x ,segmentation_y ,segmentation_z

def signal_channel_segmentation(data):

	threadhold = 0.2

	# 1) amplitude analyze
	index = []
	for i in range(len(data)):
		if abs( data[i] ) > threadhold:
			index.append(i)

	# 2)
	start_temp = 0
	end_temp = 0
	temp_segmentation = []
	for i in range(len(index)-1):

		if start_temp == 0:
			start_temp = index[i]

		if index[i+1] == index[i]+1:
			# print index[i+1]
			end_temp = index[i+1]
		elif end_temp != 0:
			temp_segmentation.append([start_temp,end_temp])
			start_temp = 0
			end_temp = 0
		else:
			start_temp = 0
			end_temp = 0
	temp_segmentation.append([start_temp,end_temp])

	return temp_segmentation

def sum_segmentation( x_data, y_data, z_data):

	extand = []

	for i in range(len(x_data)):
		extand = extand + range(x_data[i][0],x_data[i][1]+1)
	for i in range(len(y_data)):
		extand = extand + range(y_data[i][0],y_data[i][1]+1)
	for i in range(len(z_data)):
		extand = extand + range(z_data[i][0],z_data[i][1]+1)

	extand = np.array(extand)
	sort_extand = np.sort(extand)
	unique_sort_extand = np.unique(sort_extand)

	index = unique_sort_extand.tolist()

	start_temp = 0
	end_temp = 0
	temp_segmentation = []

	for i in range(len(index)-1):

		if start_temp == 0:
			start_temp = index[i]

		# if index[i+1] == index[i]+1:
		# 	# print index[i+1]
		# 	end_temp = index[i+1]
		if index[i+1]-index[i] <3:
			end_temp = index[i+1]
		elif end_temp != 0:

			if end_temp - start_temp > 20:
				temp_segmentation.append([start_temp,end_temp])
			start_temp = 0
			end_temp = 0
		else:
			start_temp = 0
			end_temp = 0

	if end_temp - start_temp > 20:
		temp_segmentation.append([start_temp,end_temp])

	return temp_segmentation

def get_segmentation_list(filter_x, filter_y, filter_z, segmentation_list):

	s_x, s_y, s_z = [],[],[]

	for i in range(len(segmentation_list)):
		start = segmentation_list[i][0]
		end = segmentation_list[i][1]
		s_x.append(filter_x[start:end])
		s_y.append(filter_y[start:end])
		s_z.append(filter_z[start:end])

	return s_x, s_y, s_z

def get_np_array(raw_x, raw_y, raw_z):

	results = []

	for i in range(len(raw_x)):
		results.append([raw_x[i],raw_y[i],raw_z[i]])

	return np.array(results)

def downsampling( length, raw_x, raw_y, raw_z ):

	sampling_number = 50

	ds_x, ds_y, ds_z = [],[],[]

	for i in range(length):
		ds_x.append(signal.resample(raw_x[i], sampling_number))
		ds_y.append(signal.resample(raw_y[i], sampling_number))
		ds_z.append(signal.resample(raw_z[i], sampling_number))

	return ds_x,ds_y,ds_z

def normalization(length,raw_x, raw_y, raw_z):

	normal = []

	for i in range(length):
		downsampling = get_np_array(raw_x[i], raw_y[i], raw_z[i])
		normal.append(preprocessing.scale(downsampling))

	np_normal = []

	for i in range(len(normal)):
		np_normal.append(np.array(normal[i]).swapaxes(0, 1))

	normal_x, normal_y, normal_z = [],[],[]
	for i in range(len(np_normal)):
		normal_x.append(np_normal[i][0])
		normal_y.append(np_normal[i][1])
		normal_z.append(np_normal[i][2])

	return normal_x, normal_y, normal_z

	# return normal[:,0],normal[:,1],normal[:,2]

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def rotation_signal(target_first_pc,train_first_pc ,train_data):

	theta = np.arccos(np.dot(target_first_pc, train_first_pc) / (np.linalg.norm(target_first_pc) * np.linalg.norm(train_first_pc)))

	axis = np.cross(target_first_pc, train_first_pc)


	n = []
	for i in range(50):
		n.append(np.dot(rotation_matrix(-axis,theta), train_data[i]))

	return np.array(n)

def rmse(target_signal, train_siganl):

	rms = math.sqrt(mean_squared_error(target_signal, train_siganl))

	return rms

def rmse_timeseries(target_signal, train_siganl):

	# print train_siganl_list.shape[0]

	x_rmse_list , y_rmse_list , z_rmse_list = [],[],[]


	for i in range(50):
		x_rmse_list.append((train_siganl[i][0] - target_signal[i][0])**2)
		y_rmse_list.append((train_siganl[i][1] - target_signal[i][1])**2)
		z_rmse_list.append((train_siganl[i][2] - target_signal[i][2])**2)
	
	return x_rmse_list, y_rmse_list, z_rmse_list

	# error = 0

	# for i in range(50):
	# 	error += (train_siganl[i:0] - target_signal[i][0])**2+ (train_siganl[i][1] - target_signal[i][1])**2 +(train_siganl[i][2] - target_signal[i][2])**2

	# error = error / 50

	# return math.sqrt(error)





