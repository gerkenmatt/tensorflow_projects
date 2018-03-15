
import os
from enum import Enum
from pathlib import Path
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
import numpy as np
import pyedflib


class States:
    rest, r_fist, l_fist, r_fist_im, l_fist_im, fists, fists_im, feet, feet_im = range(9)

class Tasks:
    T0, T1, T2 = range(3)

class RunTypes: 
	DivPatients, ShuffleAll = range(2)

state_table = np.array([[States.rest,States.rest,States.rest],   		#1
						[States.rest,States.rest,States.rest],   		#2
						[States.rest,States.l_fist, States.r_fist],		#3
						[States.rest,States.l_fist_im,States.r_fist_im],#4
						[States.rest,States.fists,States.feet],    		#5
						[States.rest,States.fists_im,States.feet_im],   #6
						[States.rest,States.l_fist, States.r_fist],		#7
						[States.rest,States.l_fist_im,States.r_fist_im],#8
						[States.rest,States.fists,States.feet], 		#9
						[States.rest,States.fists_im,States.feet_im], 	#10
						[States.rest,States.l_fist, States.r_fist],		#11
						[States.rest,States.l_fist_im,States.r_fist_im],#12
						[States.rest,States.fists,States.feet], 		#13
						[States.rest,States.fists_im,States.feet_im]]	#14
						)
def randomize_data(data, labels):
	rng_state = np.random.get_state()
	np.random.shuffle(data)
	np.random.set_state(rng_state)
	np.random.shuffle(labels)
	return data, labels


def get_eeg_data():
	rootdir = os.path.join(str(Path.home()), 'SeniorProject/EEG_Dataset/')
	num_patients = 109
	num_records = 14
	channels = [9, 11, 13]
	sig_len = 640
	sig_divs = 2

	#the completed eeg data set to be sent to Tensorflow
	eeg_data = []
	eeg_data_labels = []
	sample_nums = []
	record_nums = []
	records = {'03', '07', '11'}
	patient_nums = []

	#iterate through every volunteer's directory (S001, S002,..., S109)
	for i in range(1, num_patients  + 1):
		direc = rootdir + 'S%03d' % i
		st = "VOLUNTEER #" + '%03d' % i
		print (st)

		#iterate through the EDF num_records in the directory (S001R01.edf, S001R02.edf,..., S001R14.edf)
		for r_num in records: 
			#create new EdfReader for the EDF record
			edf_fname = 'S' + '%03d' % i + 'R' + r_num + '.edf'
			edf_fname = os.path.join(direc, edf_fname)

			edf = pyedflib.EdfReader(edf_fname)
			
			start_samples = []
			cur_tasks = []
			cur_states = []

			ann_fname = '/ann' + r_num + '.txt'
			ann_fname = direc + ann_fname

			#create array of starting samples for each task in the ann-file (0, 672, 1328,...)
			#create array of labels for each task in the ann-file (T0, T1, T2)
			with open(ann_fname) as fobj:
				for line in fobj: 
					row = line.split()
					start_samples.append(row[1])
					cur_tasks.append(row[2])

			#change the tasks to state values (rest, left_fist, right_fist,...)
			for task in cur_tasks: 
				cur_states.append(state_table[int(r_num) - 1][int(task[1])])

			#iterate through the thirty 4-second subsignals
			for k in range(len(start_samples)):

				if cur_states[k] == States.l_fist or cur_states[k] == States.r_fist:

					#divide signal further into divisions
					for div in range(sig_divs):
						div_length = int(sig_len/sig_divs)
						signal_arr = np.zeros((len(channels), div_length ))

						#iterate through all channels, save the subsignal
						for l in range(len(channels)):
							start = int(int(start_samples[k]) + div*div_length)
							signal_arr[l] = edf.readSignal(int(channels[l]), int(start_samples[k]) + div*div_length, div_length)


						#add the full signal_arr to the eeg_data array
						eeg_data.append(signal_arr)
						#add the state to the eeg_data label array
						eeg_data_labels.append(cur_states[k])
						sample_nums.append(k)
						record_nums.append(int(r_num))
						patient_nums.append(i)
					

	# print_eeg_samples(eeg_data, patient_nums, record_nums, sample_nums, eeg_data_labels)

	for i in range(len(eeg_data)):
		eeg_data[i] = eeg_data[i].flatten()
		# st = "feature #" + str(i) + " shape: " + str(eeg_data[i].shape) + "; label : " + str(eeg_data_labels[i]) + "; sample: " + str(sample_nums[i])
		st = "feature #" + str(i) + " patient: " + str(patient_nums[i]) + "; record: " +  str(record_nums[i])+ "; sample: " + str(sample_nums[i]) +  "; label : " + str(eeg_data_labels[i]) 
		print(st)


	train_patient_num = 78
	train_data = []
	train_labels = []
	eval_data = []
	eval_labels = []
	runType=RunTypes.ShuffleAll
	
	if runType == RunTypes.DivPatients:
		#separate the training and eval data by the patient numbers
		for i in range(len(eeg_data)):
			if patient_nums[i] <= train_patient_num:
				train_data.append(eeg_data[i])
				train_labels.append(eeg_data_labels[i])
			else: 
				eval_data.append(eeg_data[i])
				eval_labels.append(eeg_data_labels[i])

	elif runType == RunTypes.ShuffleAll: 
		# flatten the data into np arrays
		eeg_data = np.asarray(eeg_data)
		eeg_data_labels = np.asarray(eeg_data_labels)

		eeg_data, eeg_data_labels = randomize_data(eeg_data, eeg_data_labels)

		#separate data into training and evaluation data
		train_data = eeg_data[:7000,:]
		train_labels = eeg_data_labels[:7000]
		eval_data = eeg_data[7000:, :]
		eval_labels = eeg_data_labels[7000:]
		 

	print("training data shape: ", str(len(train_data)))
	print("eval data shape: ", str(len(eval_data)))
	train_data = np.asarray(train_data)
	train_labels = np.asarray(train_labels)
	eval_data = np.asarray(eval_data)
	eval_labels = np.asarray(eval_labels)

	train_data, train_labels = randomize_data(train_data, train_labels)
	eval_data, eval_labels = randomize_data(eval_data, eval_labels)



	return train_data, train_labels, eval_data, eval_labels

def print_eeg_samples(eeg_data, patient_nums, record_nums, sample_nums, eeg_data_labels):
	fig1 = plt.figure(1)
	i = 100 
	st = "feature #" + str(i) + "patient: " + str(patient_nums[i]) + "; record: " +  str(record_nums[i])+ "; sample: " + str(sample_nums[i]) +  "; label : " + str(eeg_data_labels[i]) 
	print (st)
	plt.plot(eeg_data[i][0], "r", eeg_data[i][1], "b", eeg_data[i][2], "g")
	fig2 = plt.figure(2)
	i = 104
	st = "feature #" + str(i) + "; patient: " + str(patient_nums[i]) + "; record: " +  str(record_nums[i])+ "; sample: " + str(sample_nums[i]) +  "; label : " + str(eeg_data_labels[i]) 
	print(st)
	plt.plot(eeg_data[i][0], "r", eeg_data[i][1], "b", eeg_data[i][2], "g")
	
	fig3 = plt.figure(3)
	i = 8000
	st = "feature #" + str(i) + "patient: " + str(patient_nums[i]) + "; record: " +  str(record_nums[i])+ "; sample: " + str(sample_nums[i]) +  "; label : " + str(eeg_data_labels[i]) 
	print (st)
	plt.plot(eeg_data[i][0], "r", eeg_data[i][1], "b", eeg_data[i][2], "g")
	fig4 = plt.figure(4)
	i = 8002
	st = "feature #" + str(i) + "patient: " + str(patient_nums[i]) + "; record: " +  str(record_nums[i])+ "; sample: " + str(sample_nums[i]) +  "; label : " + str(eeg_data_labels[i]) 
	print(st)
	plt.plot(eeg_data[i][0], "r", eeg_data[i][1], "b", eeg_data[i][2], "g")
	
	plt.show()





