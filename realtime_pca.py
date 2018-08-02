import data_collect as dc
import signal_processing as sp
import matplotlib.pyplot as plt
import database_processing as dp
import analysis
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def stander_pca():

	target_list = [1,100,207]

	tr_id = target_list[0]
	tr_data = dp.db_extract_one_signal_downsampling(tr_id)
	# plt.plot(tr_data[:,0])
	# plt.plot(tr_data[:,1])
	# plt.plot(tr_data[:,2])
	# plt.show()
	tr_pca = analysis.pca(tr_data, 3)
	tr_first_pc = tr_pca.components_[0]

	tl_id = target_list[1]
	tl_data = dp.db_extract_one_signal_downsampling(tl_id)
	tl_pca = analysis.pca(tl_data, 3)
	tl_first_pc = tl_pca.components_[0]

	tf_id = target_list[2]
	tf_data = dp.db_extract_one_signal_downsampling(tf_id)
	tf_pca = analysis.pca(tf_data, 3)
	tf_first_pc = tf_pca.components_[0]

	return tr_data, tl_data, tf_data, tr_first_pc, tl_first_pc, tf_first_pc

def pca_rotation(normalization_x,normalization_y,normalization_z,tr_first_pc, tl_first_pc, tf_first_pc):

	x, y, z = normalization_x[0],normalization_y[0],normalization_z[0]

	normalization = []
	for i in range(50):
		normalization.append([x[i],y[i],z[i]])

	normalization = np.array(normalization)

	train_data = normalization
	train_pca = analysis.pca(train_data, 3)
	train_first_pc = train_pca.components_[0]

	train_to_tr =  sp.rotation_signal(tr_first_pc,train_first_pc ,train_data)
	train_to_tl =  sp.rotation_signal(tl_first_pc,train_first_pc ,train_data)
	train_to_tf =  sp.rotation_signal(tf_first_pc,train_first_pc ,train_data)

	return train_to_tr, train_to_tl, train_to_tf

def pca_rmse(tr_data, tl_data, tf_data, train_to_tr, train_to_tl, train_to_tf):

	tr_score = sp.rmse(tr_data, train_to_tr)
	tl_score = sp.rmse(tl_data, train_to_tl)
	tf_score = sp.rmse(tf_data, train_to_tf)

	score = [tr_score, tl_score, tf_score]

	print score

	result = ""

	if score.index(min(score)) == 0:
		result = "TRIANGLE_RIGHT"
	elif score.index(min(score)) == 1:
		result = "TRIANGLE_LEFT"
	elif score.index(min(score)) == 2:
		result = "TRIANGLE_FLAT"


	print result

	return result


if __name__ == '__main__':
	
	# plt.ylim(-2,2)
	# plt.xlim(0,50)

	tr_data, tl_data, tf_data, tr_first_pc, tl_first_pc, tf_first_pc = stander_pca()

	raw_x_s, raw_y_s, raw_z_s = dc.get_rawdata_stander()
	stander_x, stander_y, stander_z = sp.get_stander(raw_x_s, raw_y_s, raw_z_s)

	count_frame = 0.0;
	count_correct = 0.0;


	# plt.figure(num = 1, figsize = (10, 3))
	# downsampling_setting = plt.subplot2grid((1,2), (0,0))
	# downsampling_setting.set_title('downsampling')
	# downsampling_setting.set_xlim(0,50)
	# downsampling_setting.set_ylim(-2,2)


	# rotation_setting = plt.subplot2grid((1,2), (0,1))
	# rotation_setting.set_title('rotation')
	# rotation_setting.set_xlim(0,50)
	# rotation_setting.set_ylim(-2,2)

	while True:

		raw_x, raw_y, raw_z = dc.get_rawdata(3)

		# plt.plt_raw(raw_x)

		mean_x, mean_y, mean_z = sp.get_mean_data(raw_x, raw_y, raw_z, stander_x, stander_y, stander_z)
		filter_x, filter_y, filter_z = sp.get_filter_data(mean_x, mean_y, mean_z)
		slope_x, slope_y, slope_z = sp.find_slope(filter_x, filter_y, filter_z)

		segmentation_list,x ,y ,z = sp.segmentation(slope_x, slope_y, slope_z)

		if len(segmentation_list) > 0:
			segmentation_x, segmentation_y, segmentation_z = sp.get_segmentation_list(filter_x, filter_y, filter_z, segmentation_list)
			downsampling_x,downsampling_y,downsampling_z = sp.downsampling(1, segmentation_x, segmentation_y, segmentation_z)
			# normalization_x,normalization_y,normalization_z = sp.normalization(1,downsampling_x,downsampling_y,downsampling_z)

			train_to_tr, train_to_tl, train_to_tf = pca_rotation(downsampling_x,downsampling_y,downsampling_z, tr_first_pc, tl_first_pc, tf_first_pc)

			result = pca_rmse(tr_data, tl_data, tf_data, train_to_tr, train_to_tl, train_to_tf)

			if result == "TRIANGLE_LEFT":
				count_correct += 1
			# else:
			# 	downsampling_setting.plot(downsampling_x[0])
			# 	downsampling_setting.plot(downsampling_y[0])
			# 	downsampling_setting.plot(downsampling_z[0])

			# 	rotation_setting.plot(train_to_tf[:,0])
			# 	rotation_setting.plot(train_to_tf[:,1])
			# 	rotation_setting.plot(train_to_tf[:,2])
			# 	plt.show()
			# 	plt.plot(downsampling_x[0])
			# 	plt.plot(downsampling_y[0])
			# 	plt.plot(downsampling_z[0])
			# 	plt.show()



			count_frame += 1
			print "-->" , count_frame
			print "The correct rate is : ", count_correct / count_frame
			print ""
		else:
			print "No Segmentation found."


		# del raw_x, raw_y, raw_z


