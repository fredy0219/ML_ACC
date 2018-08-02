#
# Author : WEIYU CHEN
# Copyright : "Copyright(C) 2018 WEIYU CHEN"
# License : Sun LAB, TNUA
# 
# Data collect from Myo sensor (acceleration), and implement into database.
# 
#
import sys
from myo import init, Hub, Feed
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pylab import *

data_collection_rate = 1/60.0

# Myo definition
init()

def get_rawdata_stander():
	
	return get_rawdata(2)

def get_rawdata(second):

	feed = Feed()
	hub = Hub()
	hub.run(1000, feed)

	raw_data = []

	# try connection
	try:
		# Start the Myo device to get data, if not succeed than return
		myo = feed.wait_for_single_device(timeout=2.0)
		if myo:
			print("-------- Start collecting data --------")

			collection_start_time = time.time()
			start_time = time.time()

			countdown = 0

			while time.time() - collection_start_time < second:
				if time.time() - start_time > data_collection_rate: # 50hz
					myo_raw_data = myo.acceleration
					raw_data.append([myo_raw_data.x,myo_raw_data.y,myo_raw_data.z])
					start_time =  time.time()

				current_time = int(time.time() - collection_start_time)
				if current_time > countdown:
					countdown = current_time
					print"%d seconds remaining..." % (second-countdown)


		else:
			print("No Myo connected after 2 seconds")
	except:
		print "Can't connect to Myo. Please check device."
	

	hub.shutdown()

	raw_x = []
	raw_y = []
	raw_z = []

	for i in range(len(raw_data)):
		raw_x.append(raw_data[i][0])
		raw_y.append(raw_data[i][1])
		raw_z.append(raw_data[i][2])

	return raw_x, raw_y, raw_z


