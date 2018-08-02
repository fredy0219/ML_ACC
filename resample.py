import database_processing as dp
import signal_processing as sp
import plt
import numpy as np

if __name__ == '__main__':

	signal_list = dp.db_extract_list_signal_downsampling("STRAIGHT_RIGHT",0)

	print signal_list

	

	# data = dp.db_extract_one_signal_normalization(295)
	# data2 = dp.db_extract_one_signal_normalization(294)
	# plt.plt_data_3d_alpha(data,data2)



