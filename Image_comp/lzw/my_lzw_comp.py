# Import necessary libraries
# Steps followed in LZW compression:
# 1. Divide the image into blocks.
# 2. Initialize a dictionary with single character entries.
# 3. For each block, encode the block using the dictionary:
#    - If the current pattern exists in the dictionary, continue.
#    - Otherwise, output the current pattern, add the new pattern to the dictionary.
# 4. Store the encoded blocks.
# 5. Calculate the compression ratio based on the size of the original and encoded image.
# 6. For decoding, initialize a dictionary with single character entries.
# 7. For each encoded block, decode using the dictionary:
#    - If the code exists in the dictionary, append the decoded pattern.
#    - Otherwise, handle the special case for missing code.
# 8. Output the decoded image and calculate SSIM and PSNR.

import cv2 as cv
import math
import numpy as np
import time
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_compression_ratio(
    encoded_image: list[list[int]],
    img_height: int,
    img_width: int,
    block_sz: int,
    max_dict_sz: int,
) -> float:
    """
    Calculates the compression ratio of the encoded image.

    Parameters:
        encoded_image (LZW encoded blocks): list[list[int]]
        img_height (height of the image): int
        img_width (width of the image): int
        block_sz (size of the blocks): int
        max_dict_sz (maximum dictionary size): int

    Returns:
        compression_ratio (compression ratio of the encoded image): float
    """
    if block_sz < 1 or block_sz > min(img_height, img_width):
        block_sz = min(img_height, img_width)

    # Calculate the padded height and width of the image
    padded_h = img_height + (block_sz - img_height % block_sz) % block_sz
    padded_w = img_width + (block_sz - img_width % block_sz) % block_sz

    # Calculate the number of bits used in the original image
    bits_in_original = padded_h * padded_w * 8

    # Calculate the number of bits used in the encoded image
    bits_in_encoded = sum(len(block) for block in encoded_image)
    bits_in_encoded *= math.ceil(math.log2(max_dict_sz))

    # Calculate the compression ratio
    compression_ratio = bits_in_original / bits_in_encoded

    return compression_ratio

def lzw_encode(
    img: np.ndarray[np.uint8], block_sz: int, max_dict_sz: int
) -> tuple[list[list[int]], int]:
    """
    Encodes a grayscale image using LZW compression.

    Parameters:
        img (grayscale image): np.ndarray[np.uint8]
        block_sz (size of the blocks): int
        max_dict_sz (maximum dictionary size): int

    Returns:
        encoded_image (LZW encoded blocks): list[list[int]]
        max_dict_used (maximum dictionary code used): int
    """
    height, width = img.shape

    if block_sz < 1 or block_sz > min(height, width):
        block_sz = min(height, width)

    # Zero padding to make the dimensions divisible by the block size
    padded_h = height + (block_sz - height % block_sz) % block_sz
    padded_w = width + (block_sz - width % block_sz) % block_sz
    padded_img = np.zeros((padded_h, padded_w), dtype=np.uint8)
    padded_img[:height, :width] = img

    # Split the image into blocks
    blocks = [
        padded_img[i : i + block_sz, j : j + block_sz]
        for i in range(0, padded_h, block_sz)
        for j in range(0, padded_w, block_sz)
    ]

    encoded_image = []
    max_dict_used = 255

    for block in blocks:
        # Initialize dictionary with single character entries
        code_dict = {chr(i): i for i in range(256)}
        encoded_block = []
        current_pattern = ""
        # Convert each block into a data sequence
        for pixel in block.flatten():
            current_pattern += chr(pixel)
            if current_pattern in code_dict:
                encoded_output = code_dict[current_pattern]  # Get the index if present in dict
            else:
                encoded_block.append(encoded_output)  # Else add it to the encoded output
                if len(code_dict) < max_dict_sz:
                    code_dict[current_pattern] = len(code_dict)  # Add new pattern to dict
                    max_dict_used = max(max_dict_used, len(code_dict) - 1)  # Update max dict used

                current_pattern = chr(pixel)
                encoded_output = code_dict[current_pattern]

        if current_pattern in code_dict:
            encoded_output = code_dict[current_pattern]

        encoded_block.append(encoded_output)
        encoded_image.append(encoded_block)

    return encoded_image, max_dict_used

def lzw_decode(
    encoded_image: list[list[int]],
    img_height: int,
    img_width: int,
    block_sz: int,
    max_dict_sz: int,
) -> np.ndarray[np.uint8]:
    """
    Decodes a grayscale image using LZW compression.

    Parameters:
        encoded_image (LZW encoded blocks): list[list[int]]
        img_height (height of the image): int
        img_width (width of the image): int
        block_sz (size of the blocks): int
        max_dict_sz (maximum dictionary size): int

    Returns:
        decoded_image (decoded image): np.ndarray[np.uint8]
    """
    if block_sz < 1 or block_sz > min(img_height, img_width):
        block_sz = min(img_height, img_width)

    padded_h = img_height + (block_sz - img_height % block_sz) % block_sz
    padded_w = img_width + (block_sz - img_width % block_sz) % block_sz

    decoded_image = np.zeros((padded_h, padded_w), dtype=np.uint8)  # Initialize decoded image
    block_index = 0

    for i in range(0, padded_h, block_sz):
        for j in range(0, padded_w, block_sz):
            decoded_block = []
            decoded = []
            code_dict = {i: [i] for i in range(256)}  # Initialize dictionary

            for code in encoded_image[block_index]:
                if code not in code_dict:
                    code_dict[code] = decoded + [decoded[0]]  # Handle special case for missing code

                decoded_block += code_dict[code]  # Append decoded sequence

                if (
                    0 < len(code_dict) < max_dict_sz
                    and decoded + [code_dict[code][0]] not in code_dict.values()
                ):
                    code_dict[len(code_dict)] = decoded + [code_dict[code][0]]  # Update dictionary

                decoded = code_dict[code]

            """
             decoded + [code_dict[code][0]] appends the first symbol of the sequence corresponding to code to the existing decoded list. 
            """

            decoded_image[i : i + block_sz, j : j + block_sz] = np.array(
                decoded_block, dtype=np.uint8
            ).reshape(block_sz, block_sz) # reshape to block size

            block_index += 1

    return decoded_image[:img_height, :img_width]

if __name__ == "__main__":
    # Read the input image in grayscale
    img_path = input("Enter the path to the image: ")
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    # Get the height and width of the input image
    height, width = img.shape

    # Get the block size for LZW encoding
    block_size = int(input("Enter the block size: "))

    # Get the maximum dictionary size for LZW encoding
    max_dict_size = int(input("Enter the maximum dictionary size: "))

    # Encode the input image using LZW encoding and save the encoded data to a file
    start = time.time()
    encoded_image, max_dict_used = lzw_encode(img, block_size, max_dict_size)
    end = time.time()
    print("Encoding time:", end - start)
    with open("lzw\\encoded_lzw.txt", "w") as file:
        file.write(f"{height} {width} {block_size}\n")
        for block in encoded_image:
            file.write(" ".join(map(str, block)) + "\n")

    # Calculate the compression ratio of the encoded image
    compression_ratio = calculate_compression_ratio(
        encoded_image, height, width, block_size, max_dict_size
    )
    print("Compression ratio:", compression_ratio)

    # Decode the encoded image using LZW decoding
    start = time.time()
    decoded_image = lzw_decode(encoded_image, height, width, block_size, max_dict_size)
    end = time.time()
    print("Decoding time:", end - start)
    cv.imwrite("lzw\\decoded_lzw.png", decoded_image)

    # Calculate SSIM and PSNR
    ssim_value = ssim(img, decoded_image)
    psnr_value = psnr(img, decoded_image)
    print("SSIM:", ssim_value)
    print("PSNR:", psnr_value)
