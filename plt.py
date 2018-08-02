import matplotlib.pyplot as plt
from matplotlib.pylab import *
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
import numpy as np

def figure_segmentation_setting():
	figure_segmentation = figure(num = 2, figsize = (21, 8))
	figure_segmentation.suptitle("Segmentation (TRIANGLE_FLAT)", fontsize=12) 
	result_segmentation_x_setting = plt.subplot2grid((12,1), (0,0),rowspan=2)
	result_segmentation_y_setting = plt.subplot2grid((12,1), (3,0),rowspan=2)
	result_segmentation_z_setting = plt.subplot2grid((12,1), (6,0),rowspan=2)
	result_segmentation_setting = plt.subplot2grid((12,1), (9,0),rowspan=2)

	result_segmentation_x_setting.set_title('Segmentation of x channel',fontsize = 10)
	result_segmentation_x_setting.set_ylim(0,5)
	result_segmentation_x_setting.set_xlim(0,2000)
	result_segmentation_x_setting.grid(True)
	result_segmentation_x_setting.set_ylabel("Amplitude")

	result_segmentation_y_setting.set_title('Segmentation of y channel',fontsize = 10)
	result_segmentation_y_setting.set_ylim(0,5)
	result_segmentation_y_setting.set_xlim(0,2000)
	result_segmentation_y_setting.grid(True)
	result_segmentation_y_setting.set_ylabel("Amplitude")

	result_segmentation_z_setting.set_title('Segmentation of z channel',fontsize = 10)
	result_segmentation_z_setting.set_ylim(0,5)
	result_segmentation_z_setting.set_xlim(0,2000)
	result_segmentation_z_setting.grid(True)
	result_segmentation_z_setting.set_ylabel("Amplitude")

	result_segmentation_setting.set_title('Collecting segmentation',fontsize = 10)
	result_segmentation_setting.set_ylim(0,5)
	result_segmentation_setting.set_xlim(0,2000)
	result_segmentation_setting.grid(True)
	result_segmentation_setting.set_ylabel("Amplitude")

	return result_segmentation_x_setting, result_segmentation_y_setting, result_segmentation_z_setting ,result_segmentation_setting

def plt_segmentation(slope_x, slope_y, slope_z,segmentation_list, segmentation_x, segmentation_y, segmentation_z, starting_id ):

	result_segmentation_x_setting, result_segmentation_y_setting, result_segmentation_z_setting ,result_segmentation_setting = figure_segmentation_setting()

	result_segmentation_x, = result_segmentation_x_setting.plot(np.arange(len(slope_x)),slope_x,'r-', label="ax", linewidth = 0.5)
	for i in range(len(segmentation_x)):
		result_segmentation_x_setting.axvspan( segmentation_x[i][0],segmentation_x[i][1],color = 'y',alpha=0.5, lw=0 )
	
	result_segmentation_y, = result_segmentation_y_setting.plot(np.arange(len(slope_y)),slope_y,'g-', label="ay",linewidth = 0.5)
	for i in range(len(segmentation_y)):
		result_segmentation_y_setting.axvspan( segmentation_y[i][0],segmentation_y[i][1],color = 'y',alpha=0.5, lw=0 )
	
	result_segmentation_z, = result_segmentation_z_setting.plot(np.arange(len(slope_z)),slope_z,'b-', label="az",linewidth = 0.5)
	for i in range(len(segmentation_z)):
		result_segmentation_z_setting.axvspan( segmentation_z[i][0],segmentation_z[i][1],color = 'y',alpha=0.5, lw=0 )
	
	result_segmentation_sum_x, = result_segmentation_setting.plot(np.arange(len(slope_z)),slope_x,'r-', label="ax",linewidth = 0.5)
	result_segmentation_sum_y, = result_segmentation_setting.plot(np.arange(len(slope_z)),slope_y,'g-', label="ay",linewidth = 0.5)
	result_segmentation_sum_z, = result_segmentation_setting.plot(np.arange(len(slope_z)),slope_z,'b-', label="az",linewidth = 0.5)
	for i in range(len(segmentation_list)):
		result_segmentation_setting.axvspan( segmentation_list[i][0],segmentation_list[i][1],color = 'y',alpha=0.5, lw=0 )
	result_segmentation_setting.legend(handles=[result_segmentation_sum_x,result_segmentation_sum_y,result_segmentation_sum_z], labels=["x channel", "y channel", "z channel"], loc="upper right")

	file_name = "photo/segmentation_continuous/segmentation_tf_" + str(starting_id) + "_" + str(starting_id+len(segmentation_list)-1)+".png"
	plt.savefig(file_name, format='png', dpi=300)
	plt.show()

def figure_rawdata_filter_slope_setting():

	# global figure_segmentation, result_filter_setting, result_slope_setting

	# Figure plot definition for segmentation data
	figure_segmentation = figure(num = 1, figsize = (21, 6))
	result_rawdata_setting = plt.subplot2grid((21,1), (0,0),rowspan=6)
	result_filter_setting = plt.subplot2grid((21,1), (8,0),rowspan=6)
	result_slope_setting = plt.subplot2grid((21,1), (16,0),rowspan=6)

	# resilt_segmentation_x_setting = plt.subplot2grid((15,1), (9,0),rowspan=1)
	# resilt_segmentation_y_setting = plt.subplot2grid((15,1), (10,0),rowspan=1)
	# resilt_segmentation_z_setting = plt.subplot2grid((15,1), (11,0),rowspan=1)
	# result_segmentation_setting = plt.subplot2grid((15,1), (12,0),rowspan=2)

	figure_segmentation.suptitle("Signal processing (TRIANGLE_FLAT)", fontsize=12)

	result_rawdata_setting.set_title('Raw data',fontsize = 10)
	result_rawdata_setting.set_ylim(-2,2)
	result_rawdata_setting.set_xlim(0,2000)
	result_rawdata_setting.grid(True)
	result_rawdata_setting.set_ylabel("Amplitude")

	result_filter_setting.set_title('Raw data after Savitzky-Golay Filter',fontsize = 10)
	result_filter_setting.set_ylim(-2,2)
	result_filter_setting.set_xlim(0,2000)
	result_filter_setting.grid(True)
	result_filter_setting.set_ylabel("Amplitude")

	result_slope_setting.set_title('Slope Distribution',fontsize = 10)
	result_slope_setting.set_ylim(0,5)
	result_slope_setting.set_xlim(0,2000)
	result_slope_setting.grid(True)
	result_slope_setting.set_ylabel("Amplitude")

	# axhline = result_slope_setting.axhline(y=200,linewidth=1.2,c="black",ls="dashed",label="Threshold")

	return result_rawdata_setting, result_filter_setting , result_slope_setting


def plt_rawdata_filter_slope(mean_x, mean_y, mean_z, filter_x, filter_y, filter_z, slope_x, slope_y, slope_z , starting_id, len_segmentation):

	result_rawdata_setting, result_filter_setting, result_slope_setting = figure_rawdata_filter_slope_setting()

	result_rawdata_x, = result_rawdata_setting.plot(np.arange(len(mean_x)),mean_x,'r-', label="ax", linewidth = 0.5)
	result_rawdata_y, = result_rawdata_setting.plot(np.arange(len(mean_y)),mean_y,'g-', label="ay",linewidth = 0.5)
	result_rawdata_z, = result_rawdata_setting.plot(np.arange(len(mean_z)),mean_z,'b-', label="az",linewidth = 0.5)

	result_rawdata_setting.legend(handles=[result_rawdata_x,result_rawdata_y,result_rawdata_z], labels=["x channel", "y channel", "z channel"], loc="upper right")

	result_filter_x, = result_filter_setting.plot(np.arange(len(filter_x)),filter_x,'r-', label="ax", linewidth = 0.5)
	result_filter_y, = result_filter_setting.plot(np.arange(len(filter_y)),filter_y,'g-', label="ay",linewidth = 0.5)
	result_filter_z, = result_filter_setting.plot(np.arange(len(filter_z)),filter_z,'b-', label="az",linewidth = 0.5)

	result_filter_setting.legend(handles=[result_filter_x,result_filter_y,result_filter_z], labels=["x channel", "y channel", "z channel"], loc="upper right")
	# for i in range(len(segmentation_list)):
	# 	result_filter_setting.axvspan( segmentation_list[i][0],segmentation_list[i][1],color = 'y',alpha=0.5, lw=0 )

	result_slope_x, = result_slope_setting.plot(np.arange(len(slope_x)),slope_x,'r-', label="ax",linewidth = 0.5)
	result_slope_y, = result_slope_setting.plot(np.arange(len(slope_y)),slope_y,'g-', label="ay",linewidth = 0.5)
	result_slope_z, = result_slope_setting.plot(np.arange(len(slope_z)),slope_z,'b-', label="az",linewidth = 0.5)

	result_slope_setting.legend(handles=[result_slope_x,result_slope_y,result_slope_z], labels=["x channel", "y channel", "z channel"], loc="upper right")

	file_name = "photo/singal_procseeing_continuous/signal_processing_tf_" + str(starting_id) + "_" + str(starting_id+len_segmentation)+".png"
	plt.savefig(file_name, format='png', dpi=300)
	plt.show()

def plt_pca(data_right_rate, data_left_rate, data_up_rate):

	print "plt_pca"

	right = np.asmatrix(data_right_rate)
	left = np.asmatrix(data_left_rate)
	up = np.asmatrix(data_up_rate)

	r_x = []
	r_y = []
	l_x = []
	l_y = []
	u_x = []
	u_y = []

	for i in range(100):
		r_x.append(right[i,0])
		r_y.append(right[i,1])
		l_x.append(left[i,0])
		l_y.append(left[i,1])
		u_x.appedn(up[i,0])
		u_y.appedn(up[i,1])



	plt.scatter(r_x,r_y)
	plt.scatter(l_x,l_y)

	plt.show()


	# x = rand(50,2)  
	# print x
	# print x[:,1]

def plt_rawdata_3d( raw_data , raw_data2):

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# ax.plot(raw_data[:,0],raw_data[:,1],raw_data[:,2], c='b')
	ax.plot(raw_data2[:,0],raw_data2[:,1],raw_data2[:,2],c='r')

	plt.show()

def plt_data_3d_alpha( data , describe="", file_name=""):

	print describe

	fig = plt.figure()

	alpha_fig = fig.add_subplot(111, projection='3d')

	alpha_fig.set_title(describe)
	alpha_fig.set_xlim(-2,2)
	alpha_fig.set_ylim(-2,2)
	alpha_fig.set_zlim(-2,2)
	alpha_fig.set_xlabel("x")
	alpha_fig.set_ylabel("y")
	alpha_fig.set_zlabel("z")

	alpha_fig.scatter(data[:,0], data[:,1], data[:,2])

	for i in range(data.shape[0]):
		alpha_fig.plot(data[i:i+2,0],data[i:i+2,1],data[i:i+2,2],alpha=float(i)/data.shape[0],c='b')

	for ii in xrange(45,360,90):
		alpha_fig.view_init(azim=ii)
		plt.savefig("photo/"+file_name+"_%d.png" % ii, format='png', dpi=300)

	plt.savefig("photo/"+file_name, format='png', dpi=300)
	plt.show()

def plt_pca_3d( data , transform_data , first_pc , describe="", file_name=""):

	print describe
	
	fig = plt.figure()
	alpha_fig = fig.add_subplot(111, projection='3d')

	alpha_fig.set_title(describe)

	alpha_fig.set_xlim(-2,2)
	alpha_fig.set_ylim(-2,2)
	alpha_fig.set_zlim(-2,2)
	alpha_fig.set_xlabel("x")
	alpha_fig.set_ylabel("y")
	alpha_fig.set_zlabel("z")

	x, y, z = [],[],[]
	for ii, jj in zip(transform_data, data):
		x.append(first_pc[0]* ii[0])
		y.append(first_pc[1]* ii[0])
		z.append(first_pc[2]* ii[0])
		
	alpha_fig.plot(x,y,z,alpha=0.5,c='r')

	alpha_fig.scatter(data[:,0], data[:,1], data[:,2])

	for i in range(data.shape[0]):
		alpha_fig.plot(data[i:i+2,0],data[i:i+2,1],data[i:i+2,2],alpha=float(i)/data.shape[0],c='b')

	plt.savefig("photo/"+file_name, format='png', dpi=300)

	for ii in xrange(45,360,90):
		alpha_fig.view_init(azim=ii)
		plt.savefig("photo/"+file_name+"_%d.png" % ii, format='png', dpi=300)

	plt.show()

def plt_line_collection(data,describe="",file_name=""):

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.set_title(describe)

	# print np.array(zip(data[:,0], data[:,1])).shape
	# line_c = LineCollection(list([data[:,0],data[:,1],data[:,2]]));
	line_c = LineCollection(list([ zip(range(0,50),data[:,0]),zip(range(0,50),data[:,1]),zip(range(0,50),data[:,2])]),colors=['r','g','b']);
	# line_c.set_alpha(0.7)
	ax.add_collection3d(line_c,zs=[1,2,3],zdir='y')

	ax.set_xlabel('Time Frame')
	ax.set_xlim3d(50, 0)
	ax.set_ylabel('')
	ax.set_ylim3d(0,4)
	ax.set_yticks([1,2,3])
	ax.set_yticklabels(['x channel','y channel','z channel'])
	ax.set_zlabel('Amplitude')
	ax.set_zlim3d(-2,2)

	plt.savefig("photo/"+file_name+".png",format='png', dpi=300)
	plt.show()




def plt_rms_score_box(data):


	# print data

	plt.boxplot(data)
	plt.show()

def plt_mse(x_mse, y_mse, z_mse, describe="", file_name=""):

	print describe

	figure_mse = figure(num = 2, figsize = (21, 8))

	figure_mse.suptitle(describe, fontsize=12) 
	result_mse_x_setting = plt.subplot2grid((12,1), (0,0),rowspan=2)
	result_mse_y_setting = plt.subplot2grid((12,1), (3,0),rowspan=2)
	result_mse_z_setting = plt.subplot2grid((12,1), (6,0),rowspan=2)
	result_mse_setting = plt.subplot2grid((12,1), (9,0),rowspan=2)

	result_mse_x_setting.set_title('MSE of x channel',fontsize = 10)
	result_mse_x_setting.set_ylim(0,5)
	# result_mse_x_setting.set_xlim(0,50)
	result_mse_x_setting.grid(True)

	result_mse_y_setting.set_title('MSE of y channel',fontsize = 10)
	result_mse_y_setting.set_ylim(0,5)
	# result_mse_y_setting.set_xlim(0,50)
	result_mse_y_setting.grid(True)

	result_mse_z_setting.set_title('MSE of z channel',fontsize = 10)
	result_mse_z_setting.set_ylim(0,5)
	# result_mse_z_setting.set_xlim(0,50)
	result_mse_z_setting.grid(True)

	result_mse_setting.set_title('Sum MSE',fontsize = 10)
	result_mse_setting.set_ylim(0,5)
	# result_mse_setting.set_xlim(0,50)
	result_mse_setting.grid(True)

	result_mse_x_setting.plot(np.arange(len(x_mse)),x_mse,'r-', label="x", linewidth = 0.5)
	result_mse_y_setting.plot(np.arange(len(y_mse)),y_mse,'g-', label="x", linewidth = 0.5)
	result_mse_z_setting.plot(np.arange(len(z_mse)),z_mse,'b-', label="x", linewidth = 0.5)

	mse_x, = result_mse_setting.plot(np.arange(len(x_mse)),x_mse,'r-', label="x", linewidth = 0.5)
	mse_y, = result_mse_setting.plot(np.arange(len(y_mse)),y_mse,'g-', label="x", linewidth = 0.5)
	mse_z, = result_mse_setting.plot(np.arange(len(z_mse)),z_mse,'b-', label="x", linewidth = 0.5)

	plt.legend(handles=[mse_x,mse_y,mse_z], labels=["x channel", "y channel", "z channel"], loc="upper right")

	plt.savefig("photo/"+file_name+".png", format='png', dpi=300)

	plt.show()

def plt_raw(data , describe="", file_name=""):

	print describe

	# plt.xlim(0,49)
	plt.title(describe)
	plt.ylim(-2,2)
	x_plot, = plt.plot(data[:,0],c='r')
	y_plot, = plt.plot(data[:,1],c='y')
	z_plot, = plt.plot(data[:,2],c='b')

	plt.legend(handles=[x_plot,y_plot,z_plot], labels=["x channel", "y channel", "z channel"], loc="upper right")
	plt.savefig("photo/"+file_name+".png", format='png', dpi=300)
	plt.show()


