import time
from PIL import Image
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def rle_encode(sequence):
    rle_sequence = []
    current_count = 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]:
            current_count += 1
        else:
            rle_sequence.append((sequence[i-1], current_count))
            current_count = 1
    rle_sequence.append((sequence[-1], current_count))  # Last run
    return rle_sequence

def calculate_compression_ratio(original_bits, compressed_bits):
    return original_bits / compressed_bits

def rle_compression_row(input_matrix):
    rle_values_row = []
    for row in input_matrix:
        row_rle = rle_encode(row)
        rle_values_row.append(row_rle)
    return rle_values_row

def rle_compression_col(input_matrix):
    rle_values_col = []
    for col in zip(*input_matrix):
        col_rle = rle_encode(col)
        rle_values_col.append(col_rle)
    return rle_values_col

def get_compressed_size(rle_values, bits_per_pixel):
    print(bits_per_pixel)
    compressed_size = 0
    for rle in rle_values:
        for value, count in rle:
            compressed_size += bits_per_pixel + count.bit_length()
    return compressed_size

def save_encoded_image(rle_values, output_path):
    # Create a flat array from RLE values
    encoded_image_array = []
    for rle in rle_values:
        for value, count in rle:
            encoded_image_array.append(value)
            encoded_image_array.append(count)

    # Convert the flat array to a NumPy array
    encoded_image_array = np.array(encoded_image_array, dtype=np.uint8)
    
    # Determine the shape to save the encoded image
    encoded_image = Image.fromarray(encoded_image_array)

    # Save the image
    encoded_image.save(output_path)

def decode_from_encoded_image_rowwise(encoded_image_path, shape):
    # Load the encoded image
    encoded_image = Image.open(encoded_image_path)
    
    # Convert the image to a NumPy array
    encoded_data = np.array(encoded_image).flatten()

    # Decode the RLE data
    rle_values = []
    for i in range(0, len(encoded_data), 2):
        value = encoded_data[i]
        count = encoded_data[i + 1]
        rle_values.append((value, count))

    # Create the decompressed matrix
    decompressed_matrix = np.zeros(shape, dtype=np.uint8)
    row_index = 0
    col_index = 0
    for value, count in rle_values:
        for _ in range(count):
            decompressed_matrix[row_index, col_index] = value
            col_index += 1
            if col_index == shape[1]:
                col_index = 0
                row_index += 1

    return decompressed_matrix

def decode_from_encoded_image_colwise(encoded_image_path, shape):
    # Load the encoded image
    encoded_image = Image.open(encoded_image_path)
    
    # Convert the image to a NumPy array
    encoded_data = np.array(encoded_image).flatten()

    # Decode the RLE data
    rle_values = []
    for i in range(0, len(encoded_data), 2):
        value = encoded_data[i]
        count = encoded_data[i + 1]
        rle_values.append((value, count))

    # Create the decompressed matrix
    decompressed_matrix = np.zeros(shape, dtype=np.uint8)
    row_index = 0
    col_index = 0
    for value, count in rle_values:
        for _ in range(count):
            decompressed_matrix[row_index, col_index] = value
            row_index += 1
            if row_index == shape[0]:
                row_index = 0
                col_index += 1

    return decompressed_matrix

def get_better_compression(input_image_path):
    # Load the input image
    input_image = Image.open(input_image_path)

    # Convert the input image to grayscale
    input_image_gray = input_image.convert("L")

    # Convert the input image to a NumPy array
    input_matrix = np.array(input_image_gray)

    # Dimensions of the input image
    m, n = input_matrix.shape

    # Perform row-wise RLE
    start_encoding = time.time()
    row_rle_values = rle_compression_row(input_matrix)
    end_encoding = time.time()
    row_encoding_time = end_encoding-start_encoding
    print("row wise rle encoding time:", row_encoding_time)


    # Perform column-wise RLE
    start_encoding = time.time()
    column_rle_values = rle_compression_col(input_matrix)
    end_encoding = time.time()
    col_encoding_time = end_encoding-start_encoding
    print("col wise rle encoding time:", col_encoding_time)

    # Determine the maximum pixel value to find the number of bits per pixel
    max_value = np.max(input_matrix)
    max_value_int = int(max_value)
    bits_per_pixel = max_value_int.bit_length()

    # Calculate compressed sizes
    total_compressed_pixels_rows = get_compressed_size(row_rle_values, bits_per_pixel)
    total_compressed_pixels_cols = get_compressed_size(column_rle_values, bits_per_pixel)

    # Calculate total original size in bits
    total_original_bits = m * n * bits_per_pixel

    # Calculate compression ratios
    compression_ratio_rows = calculate_compression_ratio(total_original_bits, total_compressed_pixels_rows)
    compression_ratio_cols = calculate_compression_ratio(total_original_bits, total_compressed_pixels_cols)

    # Determine the better compression method
    if compression_ratio_rows > compression_ratio_cols:
        better_compression = "Row-wise"
        better_compression_ratio = compression_ratio_rows
        rle_values = row_rle_values
    else:
        better_compression = "Column-wise"
        better_compression_ratio = compression_ratio_cols
        rle_values = column_rle_values

    # Save the encoded image
    save_encoded_image(rle_values, "Rle\encoded_image_rle.png")
    print("Compressed file generated as encoded_image_rle.png")
    return "Rle\encoded_image_rle.png", better_compression, better_compression_ratio

def structural_content(img1: np.ndarray, img2: np.ndarray) -> float:
    return np.sum(img1**2) / np.sum(img2**2)

def image_fidelity(img1: np.ndarray, img2: np.ndarray) -> float:
    return np.sum((img1 - img2)**2) / np.sum(img1**2)

def main_gray(input_image_path):
    # Determine the better compression method and save the encoded image
    encoded_image_path, better_compression, better_compression_ratio = get_better_compression(input_image_path)

    # Load the input image to get the original shape
    input_image = Image.open(input_image_path)
    input_image_gray = input_image.convert("L")
    input_matrix = np.array(input_image_gray)
    original_shape = input_matrix.shape

    # Decode the image from the encoded file using the better compression method
    if better_compression == "Row-wise":
        start_decoding = time.time()
        decompressed_matrix = decode_from_encoded_image_rowwise(encoded_image_path, original_shape)
        end_decoding = time.time()
        decoding_time = end_decoding-start_decoding
        print("decoding time:", decoding_time)

    else:
        start_decoding = time.time()
        decompressed_matrix = decode_from_encoded_image_colwise(encoded_image_path, original_shape)
        end_decoding = time.time()
        decoding_time = end_decoding-start_decoding
        print("decoding time:", decoding_time)
    
    # Convert the decompressed matrix to a PIL Image and save it
    decompressed_image = Image.fromarray(decompressed_matrix)
    decompressed_image.save("Rle\decompressed_rle.jpg")

    # Calculate PSNR and SSIM
    psnr_value = psnr(input_matrix, decompressed_matrix)
    ssim_value = ssim(input_matrix, decompressed_matrix)

    # Calculate additional metrics
    sc_value = structural_content(input_matrix, decompressed_matrix)
    if_value = image_fidelity(input_matrix, decompressed_matrix)
    print("Structural Content: ", sc_value)
    print("Image Fidelity: ", if_value)

    # Get the size of the original image file
    original_image_size = os.path.getsize(input_image_path)

    # Print the sizes of the original and compressed image files and the better compression method
    print("Original Image Size:", original_image_size, "bytes")
    print("Compression Ratio:", better_compression_ratio)
    print(f"Better Compression Method: {better_compression}")
    print(f"PSNR: {psnr_value} dB")
    print(f"SSIM: {ssim_value}")

if __name__ == "__main__":
    input_image_path = input("Enter input image path: ")
    main_gray(input_image_path)
