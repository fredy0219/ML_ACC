import sys
from mpl_toolkits.mplot3d import Axes3D
import database_processing as dp
import analysis
import plt
import numpy as np
from sklearn.svm import SVC
import signal_processing as sp


# model = ModelFPCA(data, for_logs=False, for_delta=False)

def get_signal_distribution( signal_list, pca_one ):
	first_pc_normal_vector = []
	second_pc_normal_vector = []

	first_pc_one = pca_one.components_[0]
	second_pc_one = pca_one.components_[1]

	for i in range(len(signal_list)):
		pca = analysis.pca(signal_list[i], 1)
		first_pc = pca.components_[0]
		cos = np.dot(first_pc,first_pc_one) / (np.sqrt((pow(first_pc[0],2)+pow(first_pc[1],2)+pow(first_pc[2],2)))
												* np.sqrt((pow(first_pc_one[0],2)+pow(first_pc_one[1],2)+pow(first_pc_one[2],2))))
		first_pc_normal_vector.append(np.rad2deg(np.arccos(cos)))

		second_pc = pca.components_[0]
		cos = np.dot(second_pc,second_pc_one) / ((abs(pow(second_pc[0],2)+abs(pow(second_pc[1],2)+abs(pow(second_pc[2],2))))) + (abs(pow(second_pc_one[0],2)+abs(pow(second_pc_one[1],2)+abs(pow(second_pc_one[2],2))))))
		second_pc_normal_vector.append(np.rad2deg(np.arccos(cos)))

	print first_pc_normal_vector

	return first_pc_normal_vector , second_pc_normal_vector

def pca_rmes(target_id, gesture):

	# data processing
	data_list = np.array(dp.db_extract_list_signal_downsampling(gesture , target_id))
	# data_list = np.array(dp.db_extract_list_signal_normalization(gesture , target_id))
	data_rms_score = []

	for i in range(data_list.shape[0]):

		train_data = data_list[i]
		train_pca = analysis.pca(train_data, 3)
		train_first_pc = train_pca.components_[0]

		train_new_data =  sp.rotation_signal(target_first_pc,train_first_pc ,train_data)
		train_new_data_pca = analysis.pca(train_new_data, 3)
		train_new_data_first_pc = train_new_data_pca.components_[0]

		data_rms_score.append( sp.rmse(target_data, train_new_data) )

	return data_rms_score



if __name__ == '__main__':

	# target_list = [1,100,227] #TRIANGLE_RIGHT,TRIANGLE_LEFT,TRIANGLE_FLAT

	# target_id = target_list[2]

	for i in range(200,299):
		target_data = dp.db_extract_one_signal_downsampling(i)
		target_pca = analysis.pca(target_data, 3)
		target_first_pc = target_pca.components_[0]


	# r_scroe = pca_rmes(target_list, target_id, 'STRAIGHT_RIGHT')
	# l_score = pca_rmes(target_list, target_id, 'STRAIGHT_LEFT')
	# plt.plt_rms_score_box([r_scroe, l_score])

		tr_score = pca_rmes(i, 'TRIANGLE_RIGHT')
		tl_score = pca_rmes(i, 'TRIANGLE_LEFT')
		tf_score = pca_rmes(i, 'TRIANGLE_FLAT')

		print i

		plt.plt_rms_score_box([tr_score,tl_score,tf_score])






	# sp.rmse_timeseries(target_data, tr_list)









