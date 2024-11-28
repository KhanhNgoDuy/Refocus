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
    