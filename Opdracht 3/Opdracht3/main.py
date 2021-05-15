from plotly import graph_objects as go
import numpy as np
from numba import cuda
from Helpers import synchronous_kernel_timeit as sync_timeit
from math import sin, cos, pi
import matplotlib.pyplot as plt

def _kz_coeffs(m, k):
    """Calc KZ coefficients. Source https://github.com/Kismuz/kolmogorov-zurbenko-filter"""

    # Coefficients at degree one
    coef = np.ones(m)
    print(coef)
    # Iterate k-1 times over coefficients
    for i in range(1, k):
        print(i)
        t = np.zeros((m, m + i * (m - 1)))
        print(t)
        for km in range(m):
            t[km, km:km + coef.size] = coef
            print(t)

        coef = np.sum(t, axis=0)
        print(coef)

    assert coef.size == k * (m - 1) + 1

    return coef / m ** k

@cuda.jit
def _calculate_intermediate_result(int_result):

    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    # Set stride equal to the number of threads we have available in either direction
    stride_x = cuda.gridDim.x * cuda.blockDim.x
    stride_y = cuda.gridDim.y * cuda.blockDim.y

    for i in range(x, _sin.shape[0], stride_x):
        for j in range(y, coeffs.size, stride_y):
            int_result[j,i] = coeffs[j]*_sin[i]

@cuda.jit
def _calculate_result_parallel(internal_result,int_result):
    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    # Set stride equal to the number of threads we have available in either direction
    stride_x = cuda.gridDim.x * cuda.blockDim.x
    stride_y = cuda.gridDim.y * cuda.blockDim.y

    coeffs_size = int_result.shape[0]

    for i in range(x +(coeffs_size - 1) // 2, int_result.shape[1]- (coeffs_size - 1) // 2, stride_x):
        # if (i < (coeffs_size-1)//2) and (i > int_result.shape[1]-(coeffs_size-1)//2):
        #     result[i] = 0
        # else:
        internal_result[i]=0
        for j in range(y, coeffs.size, stride_y):
            # result[i] += int_result[j,i-(coeffs_size-1)//2+j]
            # result[i] = result[i] + 1
            # cuda.atomic.add(result, i, 1)
            # TO DO
            # ADD ATOMIC_ADD
            cuda.atomic.add(internal_result,i,int_result[j, i - (coeffs_size - 1) // 2 + j])


def _calculate_result(input):

    coeffs_size = input.shape[0]
    result = np.zeros(input.shape[1])
    for i in range((coeffs_size - 1) // 2, input.shape[1] - (coeffs_size - 1) // 2):
        for j in range(coeffs.size):
            result[i] += input[j, i - (coeffs_size - 1) // 2 + j]
    return result

@cuda.jit
def DFT_parallel(N,samples, frequencies):
    """Execute the DFT sequentially on the GPU.

    :param samples: An array containing discrete time domain signal samples.
    :param frequencies: An empty array where the frequency components will be stored.
    """
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    #freq = frequencies[x]
    frequencies[x]= 0
    for n in range(N):
        frequencies[x] += samples[n] * (cos(2 * pi * x * n / N) - sin(2 * pi * x * n / N) * 1j)
        # frequencies[x] += sin(2 * x * n / N) * 1j

#origineel signaal aanmaken
periods = 5
N=2000
# x = np.linspace(0,2*np.pi,1024)
x = np.linspace(0,periods*2*np.pi,N)
noise = np.random.normal(0,2*np.pi,x.size)
# noise = 0
# _sin = np.sin(x)+(0.01*noise)
_sin = np.sin(x)+0.2*np.sin(25*x)

#berekenen coefficienten en gefilterd signaal
# coeffs = _kz_coeffs(20,2)
coeffs = _kz_coeffs(21,3)
int_result = np.zeros((coeffs.shape[0],_sin.shape[0]))
_calculate_intermediate_result[1,1024](int_result)
# KZ_filter = _calculate_result(int_result)
result = np.zeros(int_result.shape[1])
_calculate_result_parallel[(10,1),(100,1)](result,int_result)
# result_dev = cuda.to_device(result)
# KZ_filter = _calculate_result_parallel[1,100](result,int_result)

# t = sync_timeit( lambda: _calculate_intermediate_result[1,1024](int_result), number=10)
# print(t)

#timen berekenen gefilterd signaal zowel niet als wel met geheugen op GPU
# t = sync_timeit( lambda: _calculate_result(int_result), number=10)
# print(t)
t = sync_timeit( lambda: _calculate_result_parallel[(10,1),(100,1)](result,int_result), number=10)
print("Calculate result:")
print(t)
# result = np.zeros(int_result.shape[1])
result_dev = cuda.to_device(result)
int_result_dev = cuda.to_device(int_result)
# t = sync_timeit( lambda: _calculate_result_parallel[(10,1),(100,1)](result,int_result), number=10)
t = sync_timeit( lambda: _calculate_result_parallel[(10,1),(100,1)](result_dev,int_result_dev), number=10)
print("Calculate result on GPU memory:")
print(t)
# _calculate_result_parallel[(10,1),(100,1)](result,int_result)

#uitgang plotten
fig = go.Figure()
fig.add_traces(go.Scatter( x=x, y=_sin, name='Original Sin'))
# fig.add_traces(go.Scatter( x=x, y=KZ_filter, name='KZ_filter'))
fig.add_traces(go.Scatter( x=x, y=result, name='KZ_filter'))
fig.show()

# x_data = np.linspace(0, 2*np.pi, 100)
# y_data_1 = np.sin(x_data)
# y_data_2 = np.cos(x_data)
#
# fig = go.Figure()
# fig.add_traces(go.Scatter( x=x_data, y=y_data_1, name='Sin'))
# fig.add_traces(go.Scatter( x=x_data, y=y_data_2, name='Cos'))
# fig.show()

#berekenen DFT origineel en gefilterd signaal
SAMPLING_RATE_HZ = 100

frequencies_original = np.zeros(int(N/2+1), dtype=np.complex)
frequencies_filtered = np.zeros(int(N/2+1), dtype=np.complex)

# DFT_parallel[frequencies_original.shape[0],1](sin, frequencies_original)
DFT_parallel[frequencies_filtered.shape[0],1](N, _sin, frequencies_original)
# DFT_parallel[frequencies_filtered.shape[0],1](N,result, frequencies_filtered)
# DFT_parallel[frequencies_filtered.shape[0],1](N,result_dev, frequencies_filtered)

#timen DFT van origineel en gefilterd signaal
t = sync_timeit( lambda: DFT_parallel[frequencies_filtered.shape[0],1](N,result, frequencies_filtered), number=10)
print("Calculate DFT of result:")
print(t)
# t = sync_timeit( lambda: _calculate_result_parallel[(10,1),(100,1)](result,int_result), number=10)
frequencies_filtered_dev = cuda.to_device(frequencies_filtered)
t = sync_timeit( lambda: DFT_parallel[frequencies_filtered.shape[0],1](N,result_dev, frequencies_filtered_dev), number=10)
print("Calculate DFT of result_dev in GPU memory:")
print(t)


# print(xf.size)
#
# fig = go.Figure()
# fig.add_traces(go.Scatter( x=xf, y=abs(frequencies_original), name='Original Sin'))
# fig.add_traces(go.Scatter( x=xf, y=abs(frequencies_filtered), name='KZ_filter'))
# fig.show()

#plotten DFT origineel en gefilterd signaal
xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)

fig1, (ax1_1, ax1_2) = plt.subplots(1, 2)

# Plot the frequency components
ax1_1.plot( xf, abs(frequencies_original), color='C3' )
ax1_2.plot( xf, abs(frequencies_filtered), color='#333333' )

plt.show()

# DFT_ZK_DFT = np.zeros(int(N/2+1), dtype=np.complex)
#
# _calculate_result_parallel[(10,1),(100,1)](DFT_ZK_DFT,abs(frequencies_filtered))
#
# fig1, (ax1_1, ax1_2) = plt.subplots(1, 2)
#
# # Plot the frequency components
# ax1_1.plot( xf, abs(frequencies_filtered), color='C3' )
# ax1_2.plot( xf, abs(DFT_ZK_DFT), color='#333333' )
#
# plt.show()

# TO DO
#
# DFT-ZK-DFT