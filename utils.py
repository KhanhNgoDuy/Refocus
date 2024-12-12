import numpy as np
import math


def create_gaussian_kernel(kernel_size):
    sigma = kernel_size / 7.0
    kernel = np.zeros(shape=(kernel_size, kernel_size), dtype=np.float32)
    M = np.array([[sigma**2, 0], [0, sigma**2]])
    u = np.array([[kernel_size//2], [kernel_size//2]])
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = np.array([[i], [j]])
            g = np.exp((-1/2) * ((x-u).T @ np.linalg.inv(M) @ (x-u)))
            g = g / (2 * np.pi * (np.linalg.det(M))**0.5)
            kernel[i,j] = g
    kernel = kernel / kernel.sum()
    return np.flip(kernel)


# naive CoC kernel, not used any more
def create_soft_coc_kernel(radius, falloff=2.0):
    if radius <= 0:
        raise ValueError("Radius must be positive.")
    if falloff <= 0:
        raise ValueError("Falloff must be positive.")
    
    side = (int(2 * math.ceil(radius) + 1)) // 2
    y, x = np.ogrid[-side : side+1, -side : side+1]
    distance = np.sqrt(x**2 + y**2)
    kernel = np.maximum(0, 1 - (distance / radius)**falloff)
    kernel /= kernel.sum()
    return kernel
