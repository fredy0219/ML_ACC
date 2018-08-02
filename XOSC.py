import OSC
import time, threading, math
from functools import partial

class XOSC:

	def __init__(self):
		self.raw_x = []
		self.raw_y = []
		self.raw_z = []

	def imu(self, addr, tags, data, client_address):
		txt = "OSCMessage '%s' from %s: " % (addr, client_address)
		txt += str(data)
		
		data_str = str(data).strip('[]')
		data_list = [float(i) for i in data_str.split(', ')]

		self.raw_x.append( data_list[3] )
		self.raw_y.append( data_list[4] )
		self.raw_z.append( data_list[5] )

	def get_rawdata_stander(self):
	
		return self.get_rawdata(0.5)

	def get_rawdata(self,second):

		self.s = OSC.OSCServer(('192.168.2.101', 8000))  # listen on localhost, port 57120
		self.s.addDefaultHandlers()
		self.s.addMsgHandler('/imu', self.imu)

		self.st = threading.Thread( target = self.s.serve_forever)
		self.st.start()

		start_time = time.time()

		try :
			time.sleep(second)

		except KeyboardInterrupt :
			print "\nClosing OSCServer."
			self.s.close()
			print "Waiting for Server-thread to finish"
			self.st.join()
			print "Done"

		print "\nClosing OSCServer."
		self.s.close()
		print "Waiting for Server-thread to finish"
		self.st.join()
		print "Done"

		return self.raw_x, self.raw_y, self.raw_z

