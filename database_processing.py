import MySQLdb
import numpy as np

def db_extract_max_id():

	db = MySQLdb.connect(host="localhost",user="root", passwd="fredy0219", db="hand_action_recongnition")
	cursor = db.cursor()

	cursor.execute("SELECT MAX(ID) from raw_data")
	results = cursor.fetchall()

	id_start = 0
	if results[0][0] == None:
		id_start = 0
	else:
		id_start = results[0][0]+1

	return id_start

def db_insert_rawdata(segmentation_x, segmentation_y, segmentation_z, gesture_in):

	# print "db_insert_rawdata~~~~"

	db = MySQLdb.connect(host="localhost",user="root", passwd="fredy0219", db="hand_action_recongnition")
	cursor = db.cursor()

	cursor.execute("SELECT MAX(ID) from raw_data")
	results = cursor.fetchall()

	id_start = 0
	if results[0][0] == None:
		id_start = 0
	else:
		id_start = results[0][0]+1

	del results

	id_collect = []
	for i in range(len(segmentation_x)):
		id_collect.append(id_start+i)
		cursor.execute("INSERT INTO id_gesture(ID, Gesture) VALUES ('%d', '%s')" % (id_start + i , gesture_in))
		for j in range(len(segmentation_x[i])):
			cursor.execute( "INSERT INTO raw_data(ID, X, Y, Z) VALUES ('%d', '%f', '%f', '%f')" % (id_start + i,segmentation_x[i][j], segmentation_y[i][j], segmentation_z[i][j]))
			# print "123" 

	db.commit()
	db.close()

	return id_collect

def db_insert_filter(id_collect,filter_x, filter_y, filter_z):

	db = MySQLdb.connect(host="localhost",user="root", passwd="fredy0219", db="hand_action_recongnition")
	cursor = db.cursor()

	for i in range(len(id_collect)):
		for j in range(len(filter_x[i])):
			cursor.execute("INSERT INTO filter_data(ID, X,Y,Z) VALUES ('%d', '%f','%f','%f')" % (id_collect[i],filter_x[i][j],filter_y[i][j],filter_z[i][j]))

	db.commit()
	db.close()

	return id_collect

def db_insert_downsampling_data(id_collect,ds_x,ds_y,ds_z):

	db = MySQLdb.connect(host="localhost",user="root", passwd="fredy0219", db="hand_action_recongnition")
	cursor = db.cursor()

	for i in range(len(id_collect)):
		for j in range(len(ds_x[i])):
			# print id_collect[i],ds_x[i][j],ds_y[i][j],ds_z[i][j]
			cursor.execute("INSERT INTO downsampling_data(ID, X,Y,Z) VALUES ('%d', '%f','%f','%f')" % (id_collect[i],ds_x[i][j],ds_y[i][j],ds_z[i][j]))
	db.commit()
	db.close()

def db_insert_normalization_data(id_collect,n_x,n_y,n_z):

	db = MySQLdb.connect(host="localhost",user="root", passwd="fredy0219", db="hand_action_recongnition")
	cursor = db.cursor()

	for i in range(len(id_collect)):
		for j in range(len(n_x[i])):
			cursor.execute("INSERT INTO normalization_data(ID, X,Y,Z) VALUES ('%d', '%f','%f','%f')" % (id_collect[i],n_x[i][j],n_y[i][j],n_z[i][j]))
	db.commit()
	db.close()



def db_extract_one_signal(id):
	db = MySQLdb.connect(host="localhost",user="root", passwd="fredy0219", db="hand_action_recongnition")
	cursor = db.cursor()

	cursor.execute("SELECT * from raw_data where ID=%s" % id)
	results = cursor.fetchall()

	raw_data = []

	for row in results:
		raw_data.append([row[1],row[2],row[3]])

	return np.array(raw_data)

def db_extract_one_signal_filter(id):

	db = MySQLdb.connect(host="localhost",user="root", passwd="fredy0219", db="hand_action_recongnition")
	cursor = db.cursor()
	cursor.execute("SELECT * from filter_data where ID=%s" % id)
	results = cursor.fetchall()

	downsampling_data = []

	for row in results:
		downsampling_data.append([row[1],row[2],row[3]])

	return np.array(downsampling_data)

def db_extract_one_signal_downsampling(id):

	db = MySQLdb.connect(host="localhost",user="root", passwd="fredy0219", db="hand_action_recongnition")
	cursor = db.cursor()
	cursor.execute("SELECT * from downsampling_data where ID=%s" % id)
	results = cursor.fetchall()

	downsampling_data = []

	for row in results:
		downsampling_data.append([row[1],row[2],row[3]])

	return np.array(downsampling_data)

def db_extract_one_signal_normalization(id):

	db = MySQLdb.connect(host="localhost",user="root", passwd="fredy0219", db="hand_action_recongnition")
	cursor = db.cursor()
	cursor.execute("SELECT * from normalization_data where ID=%s" % id)
	results = cursor.fetchall()

	downsampling_data = []

	for row in results:
		downsampling_data.append([row[1],row[2],row[3]])

	return np.array(downsampling_data)



def db_extract_list_signal(gesture , target_id):

	db = MySQLdb.connect(host="localhost",user="root", passwd="fredy0219", db="hand_action_recongnition")
	cursor = db.cursor()

	cursor.execute("SELECT rf.* from raw_data AS rf INNER JOIN id_gesture AS ig ON rf.id = ig.id AND ig.Gesture = '%s'" % gesture)
	results = cursor.fetchall()

	id_list = []
	for row in results:

		if row[0] != target_id:
			id_list.append(int(row[0]))

	id_list_sort= np.unique(id_list)
	# print id_list_sort

	
	signal_list = []
	for id_current in id_list_sort:
		one_signal_list = []
		for row in results:
			if row[0] == id_current:
				one_signal_list.append(row[1:])
		signal_list.append(np.array(one_signal_list)) # add one siganl
		del one_signal_list

	db.close()

	return signal_list

def db_extract_list_signal_downsampling(gesture, target_id):

	db = MySQLdb.connect(host="localhost",user="root", passwd="fredy0219", db="hand_action_recongnition")
	cursor = db.cursor()

	cursor.execute("SELECT rf.* from downsampling_data AS rf INNER JOIN id_gesture AS ig ON rf.id = ig.id AND ig.Gesture = '%s'" % gesture)
	results = cursor.fetchall()

	id_list = []
	for row in results:
		if row[0] != target_id:
			id_list.append(int(row[0]))

	id_list_sort= np.unique(id_list)
	# print id_list_sort

	
	signal_list = []
	for id_current in id_list_sort:
		one_signal_list = []
		for row in results:
			if row[0] == id_current:
				one_signal_list.append(row[1:])
		signal_list.append(np.array(one_signal_list)) # add one siganl
		del one_signal_list

	db.close()

	return signal_list

def db_extract_list_signal_normalization(gesture, target_id):

	db = MySQLdb.connect(host="localhost",user="root", passwd="fredy0219", db="hand_action_recongnition")
	cursor = db.cursor()

	cursor.execute("SELECT rf.* from normalization_data AS rf INNER JOIN id_gesture AS ig ON rf.id = ig.id AND ig.Gesture = '%s'" % gesture)
	results = cursor.fetchall()

	id_list = []
	for row in results:
		if row[0] != target_id:
			id_list.append(int(row[0]))

	id_list_sort= np.unique(id_list)
	# print id_list_sort

	
	signal_list = []
	for id_current in id_list_sort:
		one_signal_list = []
		for row in results:
			if row[0] == id_current:
				one_signal_list.append(row[1:])
		signal_list.append(np.array(one_signal_list)) # add one siganl
		del one_signal_list

	db.close()

	return signal_list



