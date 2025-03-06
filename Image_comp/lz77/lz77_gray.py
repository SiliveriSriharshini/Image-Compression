# Steps followed in LZ77 compression:
# 1. Load the image and convert it to grayscale.
# 2. Flatten the image data into a 1D list.
# 3. Define search window and lookahead buffer sizes.
# 4. Encode the image using LZ77 encoding:
#    - Search for the longest match between the search window and the lookahead buffer.
#    - If a match is found, output the offset, length, and the next symbol.
#    - If no match is found, output 0 for offset and length, and the symbol.
# 5. Save the encoded data.
# 6. Calculate the compression ratio.
# 7. Decode the encoded data:
#    - Use the offsets and lengths to reconstruct the original data.
# 8. Save the decoded image.
# 9. Calculate and print SSIM and PSNR to evaluate the quality of the compression.

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import time

def find_longest_match(search_window, lookahead_buffer):
    max_length = 0
    start_index = 0
    # Iterate through the search window
    for i in range(len(search_window)):
        match_length = 0
        if i == len(search_window) - len(lookahead_buffer) - 1:
            break
        # Find the longest match between the search window and lookahead buffer
        for j in range(len(lookahead_buffer)):
            if i + j < len(search_window):
                if search_window[i + j] == lookahead_buffer[j]:
                    match_length += 1
                    if max_length < match_length:
                        max_length = match_length
                        start_index = i
                else:
                    break
            else:
                break
    return max_length, start_index

def lz77_encode(data, search_window_size, lookahead_buffer_size):
    offsets = []
    lengths = []
    symbols = []
    i = 0
    # Iterate through the data to encode
    while i < len(data):
        if i < 2:
            # Handle cases where data is too short to form a match
            if i == 0:
                offsets.append(0)
                lengths.append(0)
                symbols.append(data[i])
                i += 1
                continue
            else:
                if i > 0 and data[i] != data[0]:
                    offsets.append(0)
                    lengths.append(0)
                    symbols.append(data[i])
                    i += 1
                    continue
                else:
                    offsets.append(1)
                    lengths.append(1)
                    symbols.append(data[i + 1])
                    i += 2
                    continue
        else:
            lookahead_buffer = data[i:i + lookahead_buffer_size]
            search_index = min(i, search_window_size)
            search_window = data[i - search_index:i]
            result = find_longest_match(search_window + lookahead_buffer, lookahead_buffer)
            if result[0] == len(lookahead_buffer):
                if i + result[0] == len(data):
                    symbol = -1
                else:
                    symbol = data[i + lookahead_buffer_size]
            else:
                symbol = lookahead_buffer[result[0]]
            if result[0] == 0:
                offsets.append(0)
                lengths.append(0)
                symbols.append(symbol)
            else:
                offsets.append(search_index - result[1])
                lengths.append(result[0])
                symbols.append(symbol)
            i += result[0] + 1
    return offsets, lengths, symbols

def lz77_decode(offsets, lengths, symbols):
    i = 0
    decoded_output = []
    # Iterate through the encoded data to decode
    while i < len(offsets):
        if offsets[i] == 0:
            decoded_output.append(symbols[i])
        else:
            current_pos = len(decoded_output)
            for j in range(lengths[i]):
                decoded_output.append(decoded_output[current_pos - offsets[i] + j])
            decoded_output.append(symbols[i])
        i += 1
    decoded_array = np.array(decoded_output)
    return decoded_array

def load_image(image_path):
    image = Image.open(image_path).convert('L')
    return np.array(image), image.mode, image.size

def save_image(array, output_path):
    array = np.array(array, dtype=np.uint8)
    image = Image.fromarray(array)
    image.save(output_path)

def compress_image(input_path, output_path, search_window_size, lookahead_buffer_size):
    data, mode, size = load_image(input_path)
    flattened_data = data.flatten().tolist()
    offsets, lengths, symbols = lz77_encode(flattened_data, search_window_size, lookahead_buffer_size)
    
    encoded_data = []
    for i in range(len(offsets)):
        encoded_data.append(offsets[i])
        encoded_data.append(lengths[i])
        encoded_data.append(symbols[i] if symbols[i] != -1 else 0)

    encoded_array = np.array(encoded_data, dtype=np.uint16)
    save_image(encoded_array, output_path)
    
    return mode, size

def calculate_compression_ratio(encoded_offsets, encoded_lengths, encoded_symbols, original_size):
    bits_encoded = 0
    for i in range(len(encoded_offsets)):
        bits_encoded += len(bin(abs(encoded_offsets[i]))[2:])
        bits_encoded += len(bin(encoded_lengths[i])[2:])
        if encoded_symbols[i] != '0':
            bits_encoded += 8
    
    compression_ratio = original_size / bits_encoded
    return compression_ratio

if __name__ == "__main__":
    print("LZ77 Compression Algorithm")
    print("=================================================================")
    file = input("Enter the filename: ")
    image_data, mode, shape = load_image(file)
    width, height = shape
    inv_shape = (height, width)
    data_to_encode = image_data.flatten().tolist()

    search_window_size = int(input("Enter the Search Window Size: "))
    lookahead_buffer_size = int(input("Enter the Lookahead Buffer Size: "))

    # Encode the image
    start_time = time.time()
    offsets, lengths, symbols = lz77_encode(data_to_encode, search_window_size, lookahead_buffer_size)
    end_time = time.time()
    encoding_time = end_time - start_time
    print("Encoding time:", encoding_time)

    output_image_path = "lz77\\encoded_image_77.png"
    print(f"Compressed image saved as {output_image_path}")
    compress_image(file, output_image_path, search_window_size, lookahead_buffer_size)

    original_size = width * height * 8
    compression_ratio = calculate_compression_ratio(offsets, lengths, symbols, original_size)
    print("Compression ratio for LZ77:", compression_ratio)

    # Decode the image
    start_time = time.time()
    decoded_array = lz77_decode(offsets, lengths, symbols)
    decoded_array = decoded_array.reshape(inv_shape)
    output_path = "lz77\\decoded_image_77.png"
    save_image(decoded_array, output_path)
    end_time = time.time()
    decoding_time = end_time - start_time
    print("Decoding time:", decoding_time)

    print("Checking if input and output image dimensions match")
    print("Input image dimensions:", shape)
    print("Output image dimensions:", decoded_array.shape)

    # Calculate SSIM and PSNR
    ssim_value = ssim(image_data, decoded_array)
    psnr_value = psnr(image_data, decoded_array)

    print("SSIM:", ssim_value)
    print("PSNR:", psnr_value)

