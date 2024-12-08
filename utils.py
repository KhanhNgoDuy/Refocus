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


def create_soft_coc_kernel(radius, falloff=2.0):
    if radius <= 0:
        raise ValueError("Radius must be positive.")
    if falloff <= 0:
        raise ValueError("Falloff must be positive.")
    
    side = (int(2 * math.ceil(radius) + 1)) // 2
    y, x = np.ogrid[-side : side, -side : side]
    distance = np.sqrt(x**2 + y**2)
    kernel = np.maximum(0, 1 - (distance / radius)**falloff)
    kernel /= kernel.sum()
    return kernel


# def padding(input_img, kernel_size):
#     kh, kw = kernel_size
#     ph = kh // 2
#     pw = kw // 2
#     output_img = np.pad(input_img, ((ph, ph), (pw, pw), (0,0)), mode='reflect')
#     return output_img


# def decompose(kernel):
#     size_x, size_y = kernel.shape
#     center_x = size_x // 2
#     center_y = size_y // 2
#     E = kernel[center_x, center_y]
#     v = kernel[:, center_y]
#     w = kernel[center_x, :]
#     w = w / E
#     v = v / v.sum()
#     w = w / w.sum()
#     return v[:, np.newaxis], w[np.newaxis, :] # ks*1, 1*ks


# def convolution(input_img, kernel, mode):
#     # Only Gaussian filter can run 1D
#     ih, iw, ic = input_img.shape
#     # KH, KW = kernel.shape
#     output_img = np.zeros(input_img.shape)
#     if mode == '1D':
#         v, wT = decompose(kernel)
#         print(f'Decomposed kernel to 2 vectors with shapes: {v.shape, wT.shape}')
#         v = np.stack((v, v, v), axis=-1)
#         wT = np.stack((wT, wT, wT), axis=-1)
#         # Update kernel shape
#         kh, kw = v.shape[:2]
#         padded_input = padding(input_img, v.shape[:2])
#         for i in range(ih):
#             for j in range(iw):
#                 x = padded_input[i:i+kh, j:j+kw, :]
#                 o = np.sum((x * v), axis=(0, 1))
#                 output_img[i, j] = o
#         # Update kernel shape
#         kh, kw = wT.shape[:2]
#         padded_output = padding(output_img, wT.shape[:2])
#         for i in range(ih):
#             for j in range(iw):
#                 x = padded_output[i:i+kh, j:j+kw, :]
#                 o = np.sum((x * wT), axis=(0, 1))
#                 output_img[i, j] = o
#     elif mode == '2D':
#         kernel = np.stack((kernel, kernel, kernel), axis=-1)
#         kh, kw = kernel.shape[:2]
#         padded_img = padding(input_img, (kh, kw))
#         cv2.imwrite('padded_image.jpg', padded_img)
#         for i in range(ih):
#             for j in range(iw):
#                 x = padded_img[i:i+kh, j:j+kw, :]
#                 o = np.sum((x * kernel), axis=(0, 1))
#                 output_img[i, j] = o
#     else:
#         raise NotImplementedError('Only 1D or 2D convolution is available now.')
#     ############### YOUR CODE ENDS HERE #################
#     return output_img
