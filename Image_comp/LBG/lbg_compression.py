import os
import time
import numpy as np
from sklearn.cluster import KMeans
from skimage import io
from skimage.util import view_as_windows
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

def pad_image(image, block_size):
    pad_height = (block_size[0] - image.shape[0] % block_size[0]) % block_size[0]
    pad_width = (block_size[1] - image.shape[1] % block_size[1]) % block_size[1]
    padding = ((0, pad_height), (0, pad_width))
    padded_image = np.pad(image, padding, mode='constant', constant_values=0)
    return padded_image

def divide_into_blocks(image, block_size):
    blocks = view_as_windows(image, block_size, step=block_size)
    n_blocks = blocks.shape[0] * blocks.shape[1]
    d = block_size[0] * block_size[1] * image.shape[2] if len(image.shape) == 3 else block_size[0] * block_size[1]
    return blocks.reshape((n_blocks, d))

def vector_quantization_kmeans(image, block_size, n_clusters):
    blocks = divide_into_blocks(image, block_size)
    start_compression = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(blocks)
    end_compression = time.time()
    compression_time = end_compression - start_compression
    codebook = kmeans.cluster_centers_
    labels = kmeans.labels_
    return codebook, labels, compression_time

def reconstruct_image_from_blocks(blocks, labels, codebook, image_shape, block_size):
    n_blocks_row = image_shape[0] // block_size[0]
    n_blocks_col = image_shape[1] // block_size[1]
    reconstructed_image = np.zeros(image_shape, dtype=blocks.dtype)
    for i in range(n_blocks_row):
        for j in range(n_blocks_col):
            block_idx = i * n_blocks_col + j
            block = codebook[labels[block_idx]].reshape(block_size)
            reconstructed_image[i*block_size[0]:(i+1)*block_size[0], j*block_size[1]:(j+1)*block_size[1]] = block
    return reconstructed_image

def calculate_metrics(original_image, compressed_image):
    ssim_value = ssim(original_image, compressed_image, multichannel=True)
    psnr_value = psnr(original_image, compressed_image)
    return ssim_value, psnr_value

def calculate_compression_ratio(original_image, compressed_image, block_size, n_clusters):
    # Calculate the size of the original image in bytes
    original_size = original_image.size * original_image.itemsize
    
    # Divide the original image into blocks to estimate the number of blocks
    blocks = divide_into_blocks(original_image, block_size)
    
    # Calculate the size of the codebook and the size of the indices
    size_codebook = n_clusters * blocks.shape[1] * original_image.itemsize
    size_indices = len(blocks) * np.log2(n_clusters) / 8  # assuming 8 bits per index
    
    # Calculate the total compressed size
    compressed_size = size_codebook + size_indices
    
    # Calculate the compression ratio
    return original_size / compressed_size
    

def save_image_jpeg(image, filename, quality=100):
    io.imsave(filename, image, quality=quality)

def calculate_sc_if(original_image, compressed_image):
    original_squared = np.sum(np.square(original_image))
    compressed_squared = np.sum(np.square(compressed_image))
    sc = original_squared / compressed_squared
    difference_squared = np.sum(np.square(original_image - compressed_image))
    if_value = 1 - (difference_squared / original_squared)
    return sc, if_value

# Example usage
if __name__ == "__main__":
    input_img = input("Enter the input image path:")
    image = io.imread(input_img)  # Load the input image

    # Save the original image as JPEG with 100% quality
    original_jpeg_path = 'lbg\\original_image.jpg'
    save_image_jpeg(image, original_jpeg_path, quality=100)

    block_size_input = input("Enter block size (e.g., 4x4): ")
    block_size = tuple(map(int, block_size_input.split('x')))

    n_clusters = int(input("Enter the number of clusters (e.g., 256):"))  # Define the number of clusters
    
    padded_image = pad_image(image, block_size)  # Pad the image to fit block size
    
    codebook, labels, compression_time = vector_quantization_kmeans(padded_image, block_size, n_clusters)  # Compress the image
    
    blocks = divide_into_blocks(padded_image, block_size)  # Divide the padded image into blocks
    reconstructed_image = reconstruct_image_from_blocks(blocks, labels, codebook, padded_image.shape, block_size)  # Reconstruct the image
    
    # Save the reconstructed image as JPEG with 100% quality
    compressed_jpeg_path = 'lbg\\reconstructed_image.jpg'
    save_image_jpeg(reconstructed_image, compressed_jpeg_path, quality=100)
    
    # Crop the reconstructed image back to the original size
    reconstructed_image_cropped = reconstructed_image[:image.shape[0], :image.shape[1]]
    
    # Calculate metrics
    ssim_value, psnr_value = calculate_metrics(image, reconstructed_image_cropped)
    compression_ratio = calculate_compression_ratio(original_jpeg_path, compressed_jpeg_path)
    sc, if_value = calculate_sc_if(image, reconstructed_image_cropped)
    
    # Print the results
    print(f"Compression Time: {compression_time} seconds")
    print(f"SSIM: {ssim_value}")
    print(f"PSNR: {psnr_value}")
    print(f"Compression Ratio: {compression_ratio}")
    print(f"Structural Content (SC): {sc}")
    print(f"Image Fidelity (IF): {if_value}")
