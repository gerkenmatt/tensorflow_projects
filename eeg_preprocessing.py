import os
from enum import Enum
from pathlib import Path
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
from scipy.signal import convolve as sig_convolve
from scipy import signal
import pyedflib
import pywt

def eeg_fir_bandpass(eeg_data, num_channels): 
	fs = 160 
	nyq = fs / 2.0
	N  = int(len(eeg_data[0]) / num_channels)
	print("FIR Bandpass Filtering signals")

	#create FIR filter
	taps = signal.firwin(32, cutoff=[2.5/nyq, 20/nyq], window='hanning', pass_zero=False)

	#iterate through all examples in the eeg data
	for sig in eeg_data:
		#iterate through each channel
		for i in range(num_channels): 
			subsig = []
			subsig.append(sig[i*N:i*N+N])
			filtered_sig = signal.lfilter(taps, 1.0, subsig)
			# conv_result = sig_convolve(subsig, taps[np.newaxis, :], mode='valid')
			# print(filtered_sig)
			# plt.plot(filtered_sig[0])
			# plt.show()
			sig[i*N:i*N+N] = filtered_sig[0]

	print("Done filtering")
	return eeg_data


def process_data(data): 
	processed_data = []

	eeg_fir_bandpass(data, 3)
	#preprocessing for signals
	for sig in data:
		processed_signal = []

		#get the energy percentages for each channel of the signal
		ep_ch_1 = energy_percents(sig[0:320])
		ep_ch_2 = energy_percents(sig[320:640])
		ep_ch_3 = energy_percents(sig[640:960])
		processed_signal.append(ep_ch_1[3])
		processed_signal.append(ep_ch_1[4])
		processed_signal.append(ep_ch_1[5])
		processed_signal.append(ep_ch_2[3])
		processed_signal.append(ep_ch_2[4])
		processed_signal.append(ep_ch_2[5])
		processed_signal.append(ep_ch_3[3])
		processed_signal.append(ep_ch_3[4])
		processed_signal.append(ep_ch_3[5])
		processed_data.append(processed_signal)


	# print("Shape of processed data: ", len(processed_data), " x ", len(processed_data[0]))
	return processed_data

def energy_percents(sig):
	w = pywt.Wavelet('db4')
	cA6, cD6, cD5, cD4, cD3, cD2, cD1 = pywt.wavedec(sig, 'db2', mode='constant', level=6)
	e1 = sum(abs(cD1)**2)/len(cD1)
	e2 = sum(abs(cD2)**2)/len(cD2)
	e3 = sum(abs(cD3)**2)/len(cD3)
	e4 = sum(abs(cD4)**2)/len(cD4)
	e5 = sum(abs(cD5)**2)/len(cD5)
	e6 = sum(abs(cD6)**2)/len(cD6)
	et = e1 + e2 + e3 + e4 + e5 + e6
	return [round(e1/et, 3), round(e2/et, 3), round(e3/et, 3), round(e4/et,3), round(e5/et,3), round(e6/et, 3)]


def input_energy_graph(data): 
	e_band1 = []
	e_band2 = []
	e_band3 = []
	e_band4 = []
	e_band5 = []
	e_band6 = []
	e_band7 = []
	e_band8 = []
	e_band9 = []

	print("DATA SHAPE: ", data.shape)


	for i in range(int(len(data) /5)): 
		# print(i)
		e_band1.append(data[i][0])
		e_band2.append(data[i][1])
		e_band3.append(data[i][2])
		e_band4.append(data[i][3])
		e_band5.append(data[i][4])
		e_band6.append(data[i][5])
		e_band7.append(data[i][6])
		e_band8.append(data[i][7])
		e_band9.append(data[i][8])

	# e_band1.sort()
	# e_band2.sort()
	# e_band3.sort()
	# e_band4.sort()
	# e_band5.sort()
	# e_band6.sort()
	# e_band7.sort()
	# e_band8.sort()
	# e_band9.sort()
	print("printing plot")
	e_bands = e_band9 + e_band8 + e_band7 + e_band6 + e_band5 + e_band4 + e_band3 + e_band2 + e_band1 
	plt.bar(range(len(e_bands)), e_bands, align='center', alpha=0.5)
	plt.show()

'''prints plots of energy band percentages for each of the 
	motor movement states'''
def energy_band_percent_graphs(eeg_data, eeg_data_labels): 

	num_states = 2
	num_channels = 3
	eps_states = [[] for x in range(num_states)]

	print("Printing energy band percent graphs")
	print(eeg_data_labels)

	#iterate through all of the eeg signal examples
	for i in range(int(len(eeg_data)/5)):

		#get the energy percentages for each channel of the signal
		ep_ch1 = energy_percents(eeg_data[i][0:320])
		ep_ch2 = energy_percents(eeg_data[i][320:640])
		ep_ch3 = energy_percents(eeg_data[i][640:960])

		#collect the energy percent lists in the proper state
		eps_states[eeg_data_labels[i]].append([ep_ch1, ep_ch2, ep_ch3])

	print("There are ", str(len(eps_states[0])), " right hand examples")
	print("There are ", str(len(eps_states[1])), " left hand examples")

	eps_avg_states = []

	#average the energy percents for each channel of each state
	for eps_state in eps_states:

		#lists for storing the averages of each percentages for each channel
		channel_eps = []
		channel_eps.append([0] * 6) 
		channel_eps.append([0] * 6) 
		channel_eps.append([0] * 6) 

		#iterate through each example beloning to this state
		for eps_example in eps_state: 

			#iterate through each channel of the example
			for i in range(num_channels): 
				#this leaves us with the energy percent list
				eps = eps_example[i]

				#accumulate ep values 
				channel_eps[i] = [a + b for a, b in zip(channel_eps[i], eps)]

		channel_eps = [[val / len(eps_state) for val in eps] for eps in channel_eps]
		eps_avg_states.append(channel_eps)

	fig1 = plt.figure(1)
	right_plot = eps_avg_states[0][0] + eps_avg_states[0][1] + eps_avg_states[0][2]
	plt.bar(range(len(right_plot)), right_plot, align='center', alpha=0.5)
	
	fig2 = plt.figure(2)
	left_plot = eps_avg_states[1][0] + eps_avg_states[1][1] + eps_avg_states[1][2]
	plt.bar(range(len(left_plot)), left_plot, align='center', alpha=0.5)
	
	plt.show()




def eeg_fft_plot(signal): 
	N = len(signal)		#number of samples
	T = 1.0 / 160 		#sampling period
	x = np.linspace(0.0, N*T, N)
	y = signal
	yf = scipy.fftpack.fft(y)
	xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

	fig, ax = plt.subplots()
	ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
	plt.show()


def eeg_power_spectral_density_plot(sig, num_channels): 
	fs = 160
	N  = int(len(sig) / num_channels)

	for i in range(num_channels): 
		f, Pxx_den = signal.periodogram(sig[i*N:i*N+N], fs)
		Pxx_den[0] = Pxx_den[1]
		# plt.ylim([1e-3, 1e3])
		plt.semilogy(f, Pxx_den)

	plt.xlabel('frequency [Hz]')
	plt.ylabel('PSD [V**2/Hz]')
	plt.show()

def eeg_fir_bandpass_plot(sig, num_channels): 
	fs = 160
	nyq = fs / 2.0
	N  = int(len(sig) / num_channels)
	cutoff_start = 2.5
	cutoff_end   = 20

	#create FIR filter
	taps = signal.firwin(N, cutoff=[cutoff_start/nyq, cutoff_end/nyq], window='hanning', pass_zero=False)
	
	for i in range(num_channels): 
		filtered_sig = signal.lfilter(taps, 1.0, sig[i*N:i*N+N])
		f, Pxx_den = signal.periodogram(filtered_sig, fs)
		Pxx_den[0] = Pxx_den[1]
		# plt.ylim([1e-3, 1e3])
		plt.semilogy(f, Pxx_den)


	plt.xlabel('frequency [Hz]')
	plt.ylabel('PSD [V**2/Hz]')
	plt.show()

