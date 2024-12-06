import numpy as np

def create_coc_kernel(radius: float):
    if radius <= 0:
        raise ValueError("Radius must be positive.")
    size = int(2 * radius) + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    # Create a circular mask
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[mask] = 1
    kernel /= kernel.sum()
    return kernel


def create_soft_coc_kernel(radius, falloff=2.0):
    if radius <= 0:
        raise ValueError("Radius must be positive.")
    if falloff <= 0:
        raise ValueError("Falloff must be positive.")
    
    size = int(2 * radius + 1)
    y, x = np.ogrid[-radius : radius+1, -radius : radius+1]
    distance = np.sqrt(x**2 + y**2)
    kernel = np.maximum(0, 1 - (distance / radius)**falloff)
    kernel /= kernel.sum()
    return kernel
    

def create_gaussian_kernel(sigma):
    kernel_size = int(6 * sigma)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
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
