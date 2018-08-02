import signal_processing as sp
import plt
import database_processing as dp
import data_collect as dc
import analysis
import numpy as np

def draw_singal():

	raw_x_s, raw_y_s, raw_z_s = dc.get_rawdata_stander()
	stander_x, stander_y, stander_z = sp.get_stander(raw_x_s, raw_y_s, raw_z_s)

	print "Finish init"

	raw_x, raw_y, raw_z = dc.get_rawdata(10)
	mean_x, mean_y, mean_z = sp.get_mean_data(raw_x, raw_y, raw_z, stander_x, stander_y, stander_z)

	filter_x, filter_y, filter_z = sp.get_filter_data(mean_x, mean_y, mean_z)
	slope_x, slope_y, slope_z = sp.find_slope(filter_x, filter_y, filter_z)

	segmentation_list, segmentation_x, segmentation_y, segmentation_z = sp.segmentation(slope_x, slope_y, slope_z)

	plt.plt_rawdata_filter_slope(mean_x, mean_y, mean_z, filter_x, filter_y, filter_z,slope_x, slope_y, slope_z)
	plt.plt_segmentation(slope_x, slope_y, slope_z,segmentation_list, segmentation_x, segmentation_y, segmentation_z)

def pca_rotation( downsampling_x,downsampling_y,downsampling_z, tr_first_pc ):

	x, y, z = downsampling_x,downsampling_y,downsampling_z

	downsampling = []
	for i in range(50):
		downsampling.append([x[i],y[i],z[i]])

	downsampling = np.array(downsampling)

	# print downsampling
	train_data = downsampling
	train_pca = analysis.pca(train_data, 3)
	train_first_pc = train_pca.components_[0]

	# print train_data

	train_to_tr =  sp.rotation_signal(tr_first_pc,train_first_pc ,train_data)
	# train_to_tl =  sp.rotation_signal(tl_first_pc,train_first_pc ,train_data)
	# train_to_tf =  sp.rotation_signal(tf_first_pc,train_first_pc ,train_data)

	return train_to_tr
if __name__ == '__main__':

	# draw_singal()

	id = 1
	# id = 200

	# Single dataset taht after standardization
	raw_data = dp.db_extract_one_signal(id)
	plt.plt_raw(raw_data , "Single Dataset", "raw_data")

	# Single dataset that after standardization and filter
	filter_data = dp.db_extract_one_signal_filter(id)
	plt.plt_raw(filter_data , "Single Dataset(after Filter)", "filter_data")

	# Single dataset that after standardization ,filter ,resampling
	downsampling_data = dp.db_extract_one_signal_downsampling(id)
	plt.plt_raw(downsampling_data, "Single Dataset(after Filter and Resampling)","resampling_data")

	# One dataset that after segmentation and downsampling in 3D (timeseries color)
	plt.plt_data_3d_alpha(downsampling_data , "Single Dataset(after Filter and Resampling) in 3D","resampling_data_3D")
	plt.plt_line_collection(downsampling_data ,"Single Dataset Exploded View","resampling_data_exploded")
	# One dataset that after segmentation in 3D , and draw the first pc in figure
	pca = analysis.pca(downsampling_data, 3)
	first_pc = pca.components_[0]
	transform_data = pca.transform(downsampling_data)
	plt.plt_pca_3d(downsampling_data , transform_data , first_pc, "Ground Truth Dataset in 3D and PCA First Eigenvector","groundtruth_data_3D_pca")
	
	# #
	test_id = 3

	# test_data = dp.db_extract_one_signal(test_id)
	test_downsampling_data = dp.db_extract_one_signal_downsampling(test_id)

	test_pca = analysis.pca(test_downsampling_data, 3)
	test_first_pc = test_pca.components_[0]
	test_transform_data = test_pca.transform(test_downsampling_data)
	plt.plt_pca_3d(test_downsampling_data , test_transform_data , test_first_pc, "Single Dataset 3D and PCA First Eigenvector","test_data_3D_pca")
	plt.plt_line_collection(test_downsampling_data ,"Single Dataset Exploded View","test_data_exploded")
	# plt.plt_raw(test_downsampling_data, "test_downsampling_data")

	# print test_downsampling_data[:,0]
	train_test = pca_rotation(test_downsampling_data[:,0],test_downsampling_data[:,1],test_downsampling_data[:,2], first_pc)

	train_test_pca = analysis.pca(train_test, 3)
	train_test_first_pc = train_test_pca.components_[0]
	train_test_transform_data = train_test_pca.transform(test_downsampling_data)
	plt.plt_pca_3d(train_test , train_test_transform_data , train_test_first_pc, "Single Dataset 3D and PCA First Eigenvector","test_data_3D_pca_rotation")
	plt.plt_line_collection(train_test ,"Single Dataset Exploded View","test_data_rotation_exploded")
	# plt.plt_raw(train_test, "test_downsampling_data")

	x_mse, y_mse, z_mse = sp.rmse_timeseries(train_test,downsampling_data)
	plt.plt_mse(x_mse, y_mse, z_mse , "MSE","test_data_mse")


	
