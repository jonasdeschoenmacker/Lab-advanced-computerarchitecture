# This is a sample Python script.
# Frederik Callens en Jonas De Schoenmacker
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settin

import time
from numba import cuda
import numba
import numpy as np # Arrays in Python
from matplotlib import pyplot as plt # Plotting library
from math import sin, cos, pi
from Helpers import synchronous_kernel_timeit as sync_timeit

# size of image
size_of_i = 1024
size_of_j = 1024
# variables
T = 64
px = np.zeros((1024,1024))
px_seq = np.zeros((1024,1024))
px_par = np.zeros((1024,1024))
px_par_no_striding = np.zeros((1024,1024))
execution_times = []


def square_image_seq(px):
    ''' Image-generating kernel for 1024 px square images using the equation
    sequential on CPU
    :param px: array of pixels
    :return: no return
    '''
    for i in range(0,size_of_i):
        for j in range(0,size_of_j):
            px[i,j] = (sin((i*2*pi)/T)+1)*(sin((j*2*pi)/T)+1)*1/4

@cuda.jit
def square_image_parallel(px):
    ''' Image-generating kernel for 1024 px square images using the equation
    parallel on GPU using striding
    :param px: array of pixels
    :return: no return
    '''
    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    # Set stride equal to the number of threads we have available in either direction
    stride_x = cuda.gridDim.x * cuda.blockDim.x
    stride_y = cuda.gridDim.y * cuda.blockDim.y

    for i in range(x, size_of_i, stride_x):
        for j in range(y, size_of_j, stride_y):
            px[i, j] = (sin((i * 2 * pi) / T) + 1) * (sin((j * 2 * pi) / T) + 1) * 1 / 4

@cuda.jit
def square_image_parallel_no_striding(px):
    ''' Image-generating kernel for 1024 px square images using the equation
    parallel on GPU without striding
    :param px: array of pixels
    :return: no return
    '''
    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    # Connect thread's absolute position within the grid with position within the array px
    i=x
    j=y
    px[i, j] = (sin((i * 2 * pi) / T) + 1) * (sin((j * 2 * pi) / T) + 1) * 1 / 4
# # --------------------------------------------------------------------------------------------------------------------
# # Testing the methods above
# # testing sequential computation of the image and timing
# print("Sequential computation of the image timing [ms]:")
# square_image_seq(px)
# t = sync_timeit( lambda: square_image_seq(px), number=10)
# print(t)
# # Plot the image
# plt.imshow(px)
# plt.show()
#
# # testing parallel computation with striding of the image and timing
# print("parallel computation with striding of the image timing [ms]:")
# square_image_parallel[(32,32),(32,32)](px)
# t = sync_timeit( lambda: square_image_parallel[(32,32), (32,32)](px), number=10)
# print(t)
# plt.imshow(px)
# plt.show()
#
# # testing parallel computation without striding of the image and timing
# print("parallel computation without striding of the image timing [ms]:")
# square_image_parallel_no_striding[(32,32), (32,32)](px)
# t = sync_timeit( lambda: square_image_parallel_no_striding[(32,32), (32,32)](px), number=10)
# print(t)
# plt.imshow(px)
# plt.show()

# # --------------------------------------------------------------------------------------------------------------------
# # Now use only a quarter of the threads - without striding - and observe the plotted image.
# # What happens, and what would happen if you had too many threads?
# # Create a subplot
# fig1, (ax1, ax2) = plt.subplots(1, 2)
# # The first image is a baseline, this image is generated using striding
# square_image_parallel[(32,32),(32,32)](px_par)
# ax1.imshow(px_par)
# # For the second image there are a few options that can be selected below
#
# # Doesn't work, to many threads
# # dit geval werkt niet want er zal 1 dimensionaal gewerkrt worden, er zullen 256 maal 256 data punten
# # berekend worden, dit is meer dan het aantal datapunten (1024 in 1 dimensie) die moeten berekend worden
# # dit geeft een CUDA_UNKNOWN_ERROR
# # square_image_parallel_no_striding[256, 256](px_par_no_striding)
#
# # Works but only 1 coulumn is caluclated
# # hier zal de berekening wel lukken want 4 maal 256 = 1024 en dit past binnen 1 dimensie (1024)
# # er zal dus maar 1 dimensie berekend worden, we zien dit op de afbeelding (sterk ingezoomd) want enkel de eerste kolom heeft waarden
# # square_image_parallel_no_striding[4, 256](px_par_no_striding)
#
#
#
# # het tweede deel (8,16) beschrijft het aantal pixels die per thread berekend worden
# # het eerste deel (32,32) is dan het aantal blocks
# # om het totaal aantal pixels van bijvoorbeeld de x as (naar beneden) die berekend zullen worden
# # moeten 32 en 8 vermeniguldigd worden dus 256 pixels in de x as
# # A quarter of the image:
# # square_image_parallel_no_striding[(16,16),(32,32)](px_par_no_striding)
# # Other shapes
# square_image_parallel_no_striding[(32,32),(16,32)](px_par_no_striding)
#
#
#
# # Complete image (all blocks)
# # 1 thread per pixel dus dan zijn er 1024 op 1024 blocks nodig om het volledig beeld te maken
# # square_image_parallel_no_striding[(1024,1024),(1,1)](px_par_no_striding)
#
# ax2.imshow(px_par_no_striding)
# plt.show()

# # --------------------------------------------------------------------------------------------------------------------
# # Overcome the problem using striding - one thread may process multiple pixels.
# square_image_parallel[(1, 1), (1, 1)](px_par)
# plt.imshow(px_par)
# plt.show()


# # --------------------------------------------------------------------------------------------------------------------
# Evaluate how the kernel behaves when increasing the total number of threads and the amount of threads per block.

# for i in range(32):
#     t = sync_timeit( lambda: square_image_parallel[(32, 32), (1+i, 1+i)](px_par), number=10)
#     execution_times.append(t)
# print([(i+1)**2 for i in range(len(execution_times))])
# print(execution_times)
# plt.plot( [(i+1)**2 for i in range(len(execution_times))], execution_times )
# plt.grid(b=True,which='both')
# plt.yscale("log")
# plt.show()

# # --------------------------------------------------------------------------------------------------------------------
# # Plot and discuss the dependency of the computational time on the number of threads. Are there any peculiarities
# # and pitfalls one must keep in mind when running a kernel function?
# square_image_parallel[(1024, 1), (1, 1)](px_par)
# for i in range(100):
#     t = sync_timeit( lambda: square_image_parallel[(1024, 1), (1, 1+i)](px_par), number=100)
#     execution_times.append(t)
#     print(f"{i+1}\n{t}")
# plt.plot( [(i+1) for i in range(len(execution_times))], execution_times )
# plt.grid(b=True,which='both')
# plt.yscale("log")
# plt.show()

# # --------------------------------------------------------------------------------------------------------------------
# # (optional) Insert the statement histogram_out_device = cuda.to_de   vice(histogram_out)
# # before calling the kernel and this time pass it the histogram_out_device array.
# # What happens and how is it manifested on the performance?
#
# t = sync_timeit(lambda: cuda.to_device(px_par))
# print(t)
# px_par_out_device = cuda.to_device(px_par)
#
# for i in range(200):
#     t = sync_timeit( lambda: square_image_parallel[(32, 32), (1, 1+i)](px_par_out_device), number=100)
#     execution_times.append(t)
#     print(f"{i+1}\n{t}")
# plt.plot( [(i+1) for i in range(len(execution_times))], execution_times )
# plt.grid(b=True,which='both')
# plt.yscale("log")
# plt.show()

