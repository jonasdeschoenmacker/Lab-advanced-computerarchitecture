# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



# Lab 1: First application: GPUs, CUDA, and Python + Numba
# Lab advanced computer architecture
# Frederik Callens & Jonas De Schoenmacker
# MELICTes  07/03/2021

import time
from numba import cuda
import numpy as np # Arrays in Python
from matplotlib import pyplot as plt # Plotting library
from math import sin, cos, pi

def synchronous_kernel_timeit( kernel, number=1, repeat=1 ):
    """Time a kernel function while synchronizing upon every function call.

    :param kernel: Lambda function containing the kernel function (including all arguments)
    :param number: Number of function calls in a single averaging interval
    :param repeat: Number of repetitions
    :return: List of timing results or a single value if repeat is equal to one
    """

    times = []
    for r in range(repeat):
        start = time.time()
        for n in range(number):
            kernel()
            cuda.synchronize() # Do not queue up, instead wait for all previous kernel launches to finish executing
        stop = time.time()
        times.append((stop - start) / number)
    return times[0] if len(times)==1 else times


def DFT_sequential(samples, frequencies):
    """Execute the DFT sequentially on the CPU.

    :param samples: An array containing discrete time domain signal samples.
    :param frequencies: An empty array where the frequency components will be stored.
    """

    for k in range(frequencies.shape[0]):
        for n in range(N):
            frequencies[k] += samples[n] * (cos(2 * pi * k * n / N) - sin(2 * pi * k * n / N) * 1j)


#1)  Design a kernel function for calculating the DFT of a signal.
@cuda.jit
def DFT_parallel_easy(samples, frequencies):
    """Execute the DFT parallel on the GPU.

    :param samples: An array containing discrete time domain signal samples.
    :param frequencies: An empty array where the frequency components will be stored.
    """
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    #freq = frequencies[x]
    for n in range(N):
        frequencies[x] += samples[n] * (cos(2 * pi * x * n / N) - sin(2 * pi * x * n / N) * 1j)

@cuda.jit
def DFT_parallel(samples, frequenciesReal, frequenciesIm):
    """Execute the DFT parallel on the GPU.

    :param samples: An array containing discrete time domain signal samples.
    :param frequenciesReal: An empty array where the real frequency components will be stored.
    :param frequenciesIm: An empty array where the imaginairy frequency components will be stored.
    """
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    sample = samples[x]

    for k in range(frequenciesReal.shape[0]): #the shape of frequenciesReal eaquals to frequenciesIm
        #frequencies[k] += sample * (cos(2 * pi * k * x / N) - sin(2 * pi * k * x / N) * 1j)
        cuda.atomic.add(frequenciesReal, k, sample * (cos(2 * pi * k * x / N)))  # Prevent race conditions
        cuda.atomic.add(frequenciesIm, k, sample * sin(2 * pi * k * x / N))  # Prevent race conditions

# --------------------------------------------------------------------------------
# Define the sampling rate and observation time
SAMPLING_RATE_HZ = 100
TIME_S = 10 # Use only integers for correct DFT results

N = SAMPLING_RATE_HZ * TIME_S


# Define sample times
x = np.linspace(0, TIME_S, int(N), endpoint=False)

# Define a group of signals and add them together
sigs = [ np.sin(x * (2*pi) * (i+1) * 2 + i*pi/16) / (i+1) for i in range(24) ]
sig_sum = np.array(sum(sigs) / len(sigs)) + 0.05


# Initiate the empty frequency components
frequencies = np.zeros(int(N/2+1), dtype=np.complex)
frequencies_easy = np.zeros(int(N/2+1), dtype=np.complex)
frequenciesReal = np.zeros(int(N/2+1))
frequenciesIm = np.zeros(int(N/2+1))


# # Time the sequential CPU function
##@numba.jit #Adding this line doesn't work if you want to run sequential code on GPU.
# print("Time for sequential")
# DFT_sequential(sig_sum, frequencies)
# t = synchronous_kernel_timeit( lambda: DFT_sequential(sig_sum, frequencies), number=10 )
# print(t)
# print()

# # running parallel code in N threads on 1 block
# print("Time for parallel")
# DFT_parallel[1,N](sig_sum, frequenciesReal, frequenciesIm)
# t_par = synchronous_kernel_timeit( lambda: DFT_parallel[1,N](sig_sum,frequenciesReal, frequenciesIm), number=10)
# print( t_par )
# print()

# # running parallel code in one thread on 1000 blocks
# print("Time for parallel")
# DFT_parallel[1000,1](sig_sum, frequenciesReal, frequenciesIm)
# t_par = synchronous_kernel_timeit( lambda: DFT_parallel[1000,1](sig_sum,frequenciesReal, frequenciesIm), number=10)
# print( t_par )
# print()

# # running parallel_easy code in N threads on one block
# print("Time for parallel easy")
# DFT_parallel_easy[1,frequencies_easy.shape[0]](sig_sum, frequencies_easy)
# t_par_easy = synchronous_kernel_timeit( lambda: DFT_parallel_easy[1,frequencies_easy.shape[0]](sig_sum,frequencies_easy), number=10)
# print( t_par_easy)

# # running parallel_easy code in 1000 blocks on 1 thread
# print("Time for parallel easy")
# DFT_parallel_easy[1000,1](sig_sum, frequencies_easy)
# t_par_easy = synchronous_kernel_timeit( lambda: DFT_parallel_easy[1000,1](sig_sum,frequencies_easy), number=10)
# print( t_par_easy)

# # running parallel_easy code in 1 thread on N blocks
# DFT_parallel_easy[frequencies_easy.shape[0],1](sig_sum, frequencies_easy)
# t_par_easy = synchronous_kernel_timeit( lambda: DFT_parallel_easy[frequencies_easy.shape[0],1](sig_sum,frequencies_easy), number=10)
# print( t_par_easy)



# --------------------------------------------------------------------------------
# # Run full DFT, N threads
# # Reset the results and run the DFT (complex)
# # Run the DFT on CPU (seq) -> frequencies1
# frequencies1 = np.zeros(int(N/2+1), dtype=np.complex)
# DFT_sequential(sig_sum, frequencies1)
#
# # Reset 2 arrays (no complex numbers possible in cuda.atomic.add) for DFT_parallel GPU
# frequenciesReal = np.zeros(int(N/2+1))
# frequenciesIm = np.zeros(int(N/2+1))
#
# #frequencies2 = np.zeros(int(N/2+1), dtype=np.complex)
# # Run DFT_parallel GPU -> frequenciesReal,frequenciesIm
# DFT_parallel[1,N](sig_sum, frequenciesReal, frequenciesIm)
#
# # Reset array for frequencies_easy (complex)
# frequencies_easy = np.zeros(int(N/2+1), dtype=np.complex)
# # Run DFT_parallel_easy -> frequencies_easy
# DFT_parallel_easy[1,frequencies_easy.shape[0]](sig_sum, frequencies_easy)
#
# # Plot to evaluate whether the results are as expected
# fig1, (ax1_1, ax2_1) = plt.subplots(1, 2)
# fig2, (ax1_2, ax2_2) = plt.subplots(1, 2)
# fig3, (ax1_3, ax2_3) = plt.subplots(1, 2)
# # Calculate the appropriate X-axis for the frequency components
# xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)
#
# # Plot all of the signal components and their sum
# for sig in sigs:
#     ax1_1.plot( x, sig, lw=0.5, color='#333333', alpha=0.5 )
#     ax1_2.plot(x, sig, lw=0.5, color='#333333', alpha=0.5)
#     ax1_3.plot(x, sig, lw=0.5, color='#333333', alpha=0.5)
# ax1_1.plot( x, sig_sum )
# ax1_2.plot( x, sig_sum )
# ax1_3.plot( x, sig_sum )
#
# # Plot the frequency components
# ax2_1.plot( xf, abs(frequencies1), color='C3' ) # figure 1: CPU sequential
# ax2_2.plot( xf, abs(frequenciesReal - frequenciesIm*1j), color='C3' ) # figure 2: GPU parallel hard
# ax2_3.plot( xf, abs(frequencies_easy), color='C3' ) # figure 3: GPU parallel easy
#
# plt.show()




# # --------------------------------------------------------------------------------
# # Run single thread graphs
# # Reset the results and run the DFT (complex)
# # Run the DFT on CPU (seq) -> frequencies1
# frequencies1 = np.zeros(int(N/2+1), dtype=np.complex)
# DFT_sequential(sig_sum, frequencies1)
#
# # Reset 2 arrays (no complex numbers possible in cuda.atomic.add) for DFT_parallel GPU
# frequenciesReal = np.zeros(int(N/2+1))
# frequenciesIm = np.zeros(int(N/2+1))
#
# #frequencies2 = np.zeros(int(N/2+1), dtype=np.complex)
# # Run DFT_parallel GPU -> frequenciesReal,frequenciesIm
# DFT_parallel[1,25](sig_sum, frequenciesReal, frequenciesIm)
#
# # Reset array for frequencies_easy (complex)
# frequencies_easy = np.zeros(int(N/2+1), dtype=np.complex)
# # Run DFT_parallel_easy -> frequencies_easy
# DFT_parallel_easy[1,25](sig_sum, frequencies_easy)
#
# # Plot to evaluate whether the results are as expected
# fig1, (ax1_1, ax2_1) = plt.subplots(1, 2)
# fig2, (ax1_2, ax2_2) = plt.subplots(1, 2)
# fig3, (ax1_3, ax2_3) = plt.subplots(1, 2)
# # Calculate the appropriate X-axis for the frequency components
# xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)
#
# # Plot all of the signal components and their sum
# for sig in sigs:
#     ax1_1.plot( x, sig, lw=0.5, color='#333333', alpha=0.5 )
#     ax1_2.plot(x, sig, lw=0.5, color='#333333', alpha=0.5)
#     ax1_3.plot(x, sig, lw=0.5, color='#333333', alpha=0.5)
# ax1_1.plot( x, sig_sum )
# ax1_2.plot( x, sig_sum )
# ax1_3.plot( x, sig_sum )
#
# # Plot the frequency components
# ax2_1.plot( xf, abs(frequencies1), color='C3' ) # figure 1: CPU sequential
# ax2_2.plot( xf, abs(frequenciesReal - frequenciesIm*1j), color='C3' ) # figure 2: GPU parallel hard
# ax2_3.plot( xf, abs(frequencies_easy), color='C3' ) # figure 3: GPU parallel easy
#
# plt.show()

# --------------------------------------------------------------------------------
# Run 25 threads graphs
# Reset the results and run the DFT (complex)
# Run the DFT on CPU (seq) -> frequencies1
frequencies1 = np.zeros(int(N/2+1), dtype=np.complex)
DFT_sequential(sig_sum, frequencies1)

# Reset 2 arrays (no complex numbers possible in cuda.atomic.add) for DFT_parallel GPU
frequenciesReal = np.zeros(int(N/2+1))
frequenciesIm = np.zeros(int(N/2+1))

#frequencies2 = np.zeros(int(N/2+1), dtype=np.complex)
# Run DFT_parallel GPU -> frequenciesReal,frequenciesIm
DFT_parallel[1,25](sig_sum, frequenciesReal, frequenciesIm)

# Reset array for frequencies_easy (complex)
frequencies_easy = np.zeros(int(N/2+1), dtype=np.complex)
# Run DFT_parallel_easy -> frequencies_easy
DFT_parallel_easy[1,25](sig_sum, frequencies_easy)

# Plot to evaluate whether the results are as expected
fig1, (ax1_1, ax2_1) = plt.subplots(1, 2)
fig2, (ax1_2, ax2_2) = plt.subplots(1, 2)
fig3, (ax1_3, ax2_3) = plt.subplots(1, 2)
# Calculate the appropriate X-axis for the frequency components
xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)

# Plot all of the signal components and their sum
for sig in sigs:
    ax1_1.plot( x, sig, lw=0.5, color='#333333', alpha=0.5 )
    ax1_2.plot(x, sig, lw=0.5, color='#333333', alpha=0.5)
    ax1_3.plot(x, sig, lw=0.5, color='#333333', alpha=0.5)
ax1_1.plot( x, sig_sum )
ax1_2.plot( x, sig_sum )
ax1_3.plot( x, sig_sum )

# Plot the frequency components
ax2_1.plot( xf, abs(frequencies1), color='C3' ) # figure 1: CPU sequential
ax2_2.plot( xf, abs(frequenciesReal - frequenciesIm*1j), color='C3' ) # figure 2: GPU parallel hard
ax2_3.plot( xf, abs(frequencies_easy), color='C3' ) # figure 3: GPU parallel easy

plt.show()
