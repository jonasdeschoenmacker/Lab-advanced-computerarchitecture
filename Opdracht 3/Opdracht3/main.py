from plotly import graph_objects as go
import numpy as np
from numba import cuda
from Helpers import synchronous_kernel_timeit as sync_timeit

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

    for i in range(x, sin.shape[0], stride_x):
        for j in range(y, coeffs.size, stride_y):
            int_result[j,i] = coeffs[j]*sin[i]

@cuda.jit
def _calculate_result_parallel(result):
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
        for j in range(y, coeffs.size, stride_y):
            result[i] += int_result[j,i-(coeffs_size-1)//2+j]
            # result[i] = result[i] + 1
            # cuda.atomic.add(result, i, 1)
            # TO DO
            # ADD ATOMIC_ADD


def _calculate_result(input):

    coeffs_size = input.shape[0]
    result = np.zeros(input.shape[1])
    for i in range((coeffs_size - 1) // 2, input.shape[1] - (coeffs_size - 1) // 2):
        for j in range(coeffs.size):
            result[i] += input[j, i - (coeffs_size - 1) // 2 + j]
    return result

periods = 5

# x = np.linspace(0,2*np.pi,1024)
x = np.linspace(0,periods*2*np.pi,1000)
noise = np.random.normal(0,2*np.pi,x.size)
sin = np.sin(x)+(0.01*noise)

# coeffs = _kz_coeffs(20,2)
coeffs = _kz_coeffs(3,3)
int_result = np.zeros((coeffs.shape[0],sin.shape[0]))
_calculate_intermediate_result[1,1024](int_result)
# KZ_filter = _calculate_result(int_result)
result = np.zeros(int_result.shape[1])
KZ_filter = _calculate_result_parallel[1,100](result)

# t = sync_timeit( lambda: _calculate_intermediate_result[1,1024](int_result), number=10)
# print(t)

# t = sync_timeit( lambda: _calculate_result(int_result), number=10)
# print(t)
# t = sync_timeit( lambda: _calculate_result_parallel[1,1024](int_result), number=10)
# t = sync_timeit( lambda: _calculate_result_parallel[1,100](result), number=10)
# print(t)
_calculate_result_parallel[(10,1),(100,1)](result)

fig = go.Figure()
fig.add_traces(go.Scatter( x=x, y=sin, name='Original Sin'))
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



# TO DO
# ADD ATOMIC_ADD