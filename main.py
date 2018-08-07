#import data_collect as dc
import XOSC
import signal_processing as sp
import plt
import database_processing as dp


if __name__ == '__main__':

	xosc = XOSC.XOSC()

	raw_x_s, raw_y_s, raw_z_s = xosc.get_rawdata_stander()
	stander_x, stander_y, stander_z = sp.get_stander(raw_x_s, raw_y_s, raw_z_s)

	print "Finish init"

	raw_x, raw_y, raw_z = xosc.get_rawdata(20)
	mean_x, mean_y, mean_z = sp.get_mean_data(raw_x, raw_y, raw_z, stander_x, stander_y, stander_z)

	filter_x, filter_y, filter_z = sp.get_filter_data(mean_x, mean_y, mean_z)
	slope_x, slope_y, slope_z = sp.find_slope(filter_x, filter_y, filter_z)

	segmentation_list, segmentation_list_x, segmentation_list_y, segmentation_list_z = sp.segmentation(slope_x, slope_y, slope_z)

	starting_id = dp.db_extract_max_id()
	plt.plt_rawdata_filter_slope(mean_x, mean_y, mean_z, filter_x, filter_y, filter_z,slope_x, slope_y, slope_z , starting_id, len(segmentation_list)-1)
	plt.plt_segmentation(slope_x, slope_y, slope_z,segmentation_list, segmentation_list_x, segmentation_list_y, segmentation_list_z , starting_id)
	

	choose = raw_input("Insert into database? (y/n)")


	if choose == 'y':

		# Get the segmentaion of data(raw data after standardization)
		raw_segmentation_x, raw_segmentation_y, raw_segmentation_z = sp.get_segmentation_list(mean_x, mean_y, mean_z, segmentation_list)
		id_collect = dp.db_insert_rawdata(raw_segmentation_x, raw_segmentation_y, raw_segmentation_z,'TRIANGLE_FLAT')

		# Get the segmentation of filter data(raw data after standardization and filter)
		filter_segmentation_x, filter_segmentation_y, filter_segmentation_z = sp.get_segmentation_list(filter_x, filter_y, filter_z, segmentation_list)
		dp.db_insert_filter(id_collect,filter_segmentation_x, filter_segmentation_y, filter_segmentation_z)
		
		# Down sampling
		downsampling_x,downsampling_y,downsampling_z = sp.downsampling( len(id_collect), filter_segmentation_x, filter_segmentation_y, filter_segmentation_z)
		dp.db_insert_downsampling_data(id_collect, downsampling_x,downsampling_y,downsampling_z)

		# # Normalization
		# normalization_x,normalization_y,normalization_z = sp.normalization(len(id_collect),downsampling_x,downsampling_y,downsampling_z)
		# dp.db_insert_normalization_data(id_collect,normalization_x,normalization_y,normalization_z)

	else:
		print "no insert"



