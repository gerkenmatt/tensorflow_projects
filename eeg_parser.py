import os
from enum import Enum
from pathlib import Path
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
from scipy import signal
import pyedflib
import random


class States:
    rest, r_fist, l_fist, r_fist_im, l_fist_im, fists, fists_im, feet, feet_im = range(9)

class Tasks:
    T0, T1, T2 = range(3)

#structure to store signal information
class EEG_Sample: 
	def __init__(self, signal, label, patient_num, record_num, sample_num): 
		self.signal = signal
		self.label = label
		self.patient_num = patient_num
		self.record_num = record_num 
		self.sample_num = sample_num

	def print_sample_info(self): 
		print("signal shape: ", self.signal.shape, " ;label: ", self.label, " ;patient: ", self.patient_num, " ;record: ", self.record_num, " ;sample: ", self.sample_num)


#maps the record number and task type (T0, T1, T2) to the State
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

#returns array of EEG_Samples
def get_eeg_samples(db_dir_name, num_patients=109, num_records=14, channels=[9,11,13], sig_len=640, num_segs=2):
	# rootdir = os.path.join(str(Path.home()), 'SeniorProject/EEG_Dataset/')
	rootdir = os.path.join(str(Path.home()), db_dir_name)


	#the completed eeg data set to be sent to Tensorflow
	eeg_samples = []
	records = {'03', '07', '11'}	#these are the records than contain the left/right fist data

	#iterate through every volunteer's directory (S001, S002,..., S109)
	for patient in range(1, num_patients  + 1):
		direc = rootdir + 'S%03d' % patient
		st = "VOLUNTEER #" + '%03d' % patient
		print (st)

		#iterate through the EDF num_records in the directory (S001R01.edf, S001R02.edf,..., S001R14.edf)
		for record_num in records: 
			#create new EdfReader for the EDF record file
			edf_reader = create_edf_reader(direc, patient, record_num)

			#get the list of [starting values, states] for each task-sample
			record_data = parse_record_ann(direc, record_num)

			#iterate through the 4 second task-samples
			for sample_num in range(len(record_data)):
				start_sample = int(record_data[sample_num][0])
				sample_sig = read_eeg_signal(edf_reader, channels, start_sample, sig_len)
				segments = segment_signal(sample_sig, num_segs, sig_len)

				for segment in segments: 
					# segment = segment.flatten()
					segment_data = EEG_Sample(segment, int(record_data[sample_num][1]), patient, record_num, sample_num)					
					eeg_samples.append(segment_data)
					# segment_data.print_sample_info()

	return eeg_samples


def train_test_split_shuffle(eeg_samples, total_samples_used, train_test_percent=0.7):

	#shuffle all samples
	random.shuffle(eeg_samples)

	#shorten the list of samples
	eeg_samples = eeg_samples[:total_samples_used]
	# eeg_data = np.asarray([x.signal for x in eeg_samples])
	# eeg_labels = np.asarray([x.label - 1 for x in eeg_samples])

	# #split the list
	# train_test_split = np.random.rand(len(eeg_samples)) < train_test_percent
	# train_data = eeg_data[train_test_split]
	# train_labels = eeg_labels[train_test_split]
	# test_data = eeg_data[~train_test_split]
	# test_labels = eeg_labels[~train_test_split]

	train_test_split = np.random.rand(len(eeg_samples)) < train_test_percent
	train_samples = eeg_samples[train_test_split]
	test_samples = eeg_samples[~train_test_split]

	# return train_data, train_labels, test_data, test_labels
	return train_samples, test_samples

def train_test_split_patients(eeg_samples, patient_div): 
	train_data = []
	train_labels = []
	test_data = []
	test_labels = []

	for sample in eeg_samples: 
		if sample.patient_num <= patient_div: 
			train_data.append(sample.signal)
			train_labels.append(sample.label)
		else: 
			test_data.append(sample.signal)
			test_labels.append(sample.label)

	train_data, train_labels = randomize_data(train_data, train_labels)
	test_data, test_labels = randomize_data(test_data, test_labels)
	return np.asarray(train_data), np.asarray(train_labels), np.asarray(test_data), np.asarray(test_labels)

def randomize_data(data, labels):
	rng_state = np.random.get_state()
	np.random.shuffle(data)
	np.random.set_state(rng_state)
	np.random.shuffle(labels)
	return data, labels

def static_randomize_data(data, labels): 
	np.random.seed(69)
	rlist = np.random.permutation(len(data))
	new_data = []
	new_labels = []

	for idx in rlist: 
		new_data.append(data[idx])
		new_labels.append(labels[idx])

	return np.asarray(new_data), np.asarray(new_labels)


#create an pyedflib.EdfReader for the given record number
def create_edf_reader(direc, patient_num, record_num): 
	edf_fname = 'S' + '%03d' % patient_num + 'R' + record_num + '.edf'
	edf_fname = os.path.join(direc, edf_fname)
	
	return pyedflib.EdfReader(edf_fname)


# parses through the ann.txt file for the given record number
# returns array of entries: [starting_sample_number, state_type]
def parse_record_ann(direc, record_num): 
	record_data = []
	ann_fname = direc + '/ann' + record_num + '.txt'

	#create array of starting samples for each task(T0, T1, T2) in the ann-file (i.e sample# 0, 672, 1328,...)
	#create array of labels for each task in the ann-file (T0, T1, T2)
	with open(ann_fname) as fobj:
		for line in fobj: 
			row = line.split()
			start_sample = row[1]
			state_type = state_table[int(record_num) - 1][int(row[2][1])]
			record_data.append([start_sample, state_type])

	# for i in range(len(record_data)): 
	# 	if record_data[i][1] != States.l_fist and record_data[i][1] == States.r_fist: 

	for sample_data in record_data: 
		if sample_data[1] != States.l_fist and sample_data[1] != States.r_fist: 
			record_data.remove(sample_data)

	return record_data


#returns array of signal values from EDF file
def read_eeg_signal(edf_reader, channels, start_sample, sig_len): 

	#create empty numpy array of shape: [channel_num, signal_length]
	signal_arr = np.zeros((len(channels), sig_len))

	#iterate through all channels
	for l in range(len(channels)):
		signal_arr[l] = edf_reader.readSignal(int(channels[l]),start_sample, sig_len)

	return signal_arr


# divide input signal into equal length segments
# returns array of shape: [total_segments]
def segment_signal(sig, num_segs, sig_len): 
	segment_len = int(sig_len / num_segs)
	segments = []

	for s in range(num_segs):
		start = s * segment_len
		end = start + segment_len
		segment = sig[:,start:end]
		segments.append(segment)

	segments = np.array(segments)
	# print("Signal Shape: ", sig.shape)
	# print("Segments Shape: ", segments.shape)
	return segments









