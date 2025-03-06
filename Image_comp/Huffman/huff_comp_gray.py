import heapq
import time
import numpy as np
from PIL import Image
from collections import Counter
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

class Node:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
        self.huff = ''

    def __lt__(self, nxt):
        return self.freq < nxt.freq

def calculate_frequency(data):
    return Counter(data)

def build_huffman_tree(frequency):
    heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
    heapq.heapify(heap) #converts list into minheap

    """
    It uses a min-heap (heap) to merge nodes representing symbols based on their frequencies. 
    The algorithm repeatedly pops the two nodes with the smallest frequencies (lo and hi) from the heap,
    assigns binary codes '0' and '1' to their symbol-node pairs, and then merges them into a new node. 
    This process continues until only one node remains in the heap, which represents the root of the Huffman tree. 
    Huffman coding ensures that more frequent symbols receive shorter binary codes, 
    achieving efficient compression suitable for various data types.
    """
    
    while len(heap) > 1:
        lo = heapq.heappop(heap) #pop the least two frquent elements
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1] #left gets zero assigned
        for pair in hi[1:]:
            pair[1] = '1' + pair[1] #right gets 1 assigned
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

        """
        len(p[-1]): Sorts primarily by the length of the binary code (p[-1]), ensuring shorter codes appear first.
        p: Secondary sorting criterion ensures stability by sorting by the pair itself if lengths are equal.
        """

    
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def huffman_encoding(data):
    if not data: #if no data return empty
        return "", {}
    
    frequency = calculate_frequency(data)
    huffman_tree = build_huffman_tree(frequency)
    
    huff_dict = {symbol: code for symbol, code in huffman_tree} #for each symbol codeword is added
    encoded_data = ''.join(huff_dict[symbol] for symbol in data)
    
    return encoded_data, huff_dict

def huffman_decoding(encoded_data, huff_dict, original_last_pixel):
    if not encoded_data or not huff_dict:
        return ""
    
    reverse_huff_dict = {v: k for k, v in huff_dict.items()} #symbol:code
    current_code = ""
    decoded_data = []
    
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_huff_dict:
            decoded_data.append(reverse_huff_dict[current_code])
            current_code = ""
    
    # Check if the last pixel is missing and add it if necessary
    if len(decoded_data) < len(original_last_pixel):
        decoded_data.append(original_last_pixel[-1])
    
    return decoded_data

def write_bitstring_to_binary(encoded_data, filepath):
    with open(filepath, "wb") as output:
        # Pad the encoded data to ensure it is a multiple of 8 bits
        encoded_data += '0' * (8 - len(encoded_data) % 8) 
        for i in range(0, len(encoded_data), 8):
            byte = encoded_data[i:i+8]
            output.write(bytes([int(byte, 2)]))

def read_bitstring_from_binary(filepath):
    with open(filepath, "rb") as file:
        byte_array = file.read()
        encoded_data = ''.join(format(byte, '08b') for byte in byte_array)
    return encoded_data.rstrip('0')  # Remove any padding added during writing

def main():
    input_image_path = input("Enter file input:")
    binary_file_path = "Huffman\\encoded_sequence.bin"
    decoded_image_path = "Huffman\\decoded_image.png"

    # Read the input image
    original_image = np.array(Image.open(input_image_path))

    # Convert image to pixel list
    pixels = original_image.flatten().tolist()

    # Huffman encoding
    start_time = time.time()
    encoded_data, huff_dict = huffman_encoding(pixels)
    encoding_time = time.time() - start_time

    # Save encoded data to binary file
    write_bitstring_to_binary(encoded_data, binary_file_path)

    # Read encoded data from binary file
    encoded_data_from_binary = read_bitstring_from_binary(binary_file_path)

    # Decode Huffman-encoded data
    start_time = time.time()
    decoded_pixels = huffman_decoding(encoded_data_from_binary, huff_dict, pixels)
    decoding_time = time.time() - start_time

    # Convert decoded symbols back to pixel values
    decoded_image_pixels = [int(pixel) for pixel in decoded_pixels]

    # Reshape the pixel list to match the original image shape
    decoded_image_array = np.array(decoded_image_pixels, dtype=np.uint8)

    # Reshape the pixel array to match the original image shape
    decoded_image_array = decoded_image_array.reshape(original_image.shape)

    # Save decoded image
    Image.fromarray(decoded_image_array).save(decoded_image_path)

    # Calculate SSIM and PSNR
    ssim_value = ssim(original_image, decoded_image_array, win_size=7, channel_axis=-1, multichannel=True)
    psnr_value = psnr(original_image, decoded_image_array)

    # Calculate compression ratio
    original_size_bits = len(pixels) * 8  # 8 bits per pixel
    compressed_size_bits = len(encoded_data)
    compression_ratio = original_size_bits / compressed_size_bits if compressed_size_bits > 0 else 0

    # Print results
    print("Encoding time:", encoding_time, "seconds")
    print("Decoding time:", decoding_time, "seconds")
    print("SSIM:", ssim_value)
    print("PSNR:", psnr_value, "dB")
    print("Compression Ratio:", compression_ratio)

if __name__ == "__main__":
    main()
