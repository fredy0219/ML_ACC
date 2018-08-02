import plt
import database_processing as dp

if __name__ == '__main__':

	data = dp.db_extract_one_signal(222)
	data2 = dp.db_extract_one_signal(223)

	plt.plt_rawdata_3d( data , data2)