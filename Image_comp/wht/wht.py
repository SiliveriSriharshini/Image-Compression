import time
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2 as cv
from skimage import io

# Define the quantization matrix
quant_matrix = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)

# Function to compute PSNR between two images
def compute_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
    mean_square_error = np.mean(np.square(image1 - image2))
    peak_snr = 20 * np.log10(255 / np.sqrt(mean_square_error))
    return peak_snr

# Function to count non-zero elements in all blocks
def count_elements(all_blocks: list[np.ndarray]) -> int:
    all_elements = 0
    for each_block in all_blocks:
        all_elements += np.trim_zeros(each_block, "b").size
    return all_elements

# Function to calculate total elements in all blocks for both grayscale and color images
def total_elements_in_blocks(
    blocks: list[np.ndarray]
    | tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]],
    is_color: bool,
) -> int:
    all_elements = 0
    if is_color:
        all_elements = (
            count_elements(blocks[0])    # channel 1
            + count_elements(blocks[1])  # channel 2
            + count_elements(blocks[2])  # channel 3
        )
    else:
        all_elements = count_elements(blocks)   # for grayscale
    return all_elements

# Function to perform zigzag scan on a block
def perform_zigzag_scan(each_block: np.ndarray) -> np.ndarray:
    each_block_shape = each_block.shape[0]
    array_zigzag = np.concatenate(
        [
            np.diagonal(each_block[::-1, :], i)[:: (2 * (i % 2) - 1)]
            for i in range(1 - each_block_shape, each_block_shape)
        ]
    )
    return array_zigzag

# Function to perform zigzag unscan on an array
def perform_zigzag_unscan(
    zigzag_array: np.ndarray, block_size: int
) -> np.ndarray:
    unscanned_array = np.zeros((block_size, block_size), dtype=np.float32)
    row, col = 0, 0
    for num in zigzag_array:
        unscanned_array[row, col] = num
        if (row + col) % 2 == 0:
            if col == block_size - 1:
                row += 1
            elif row == 0:
                col += 1
            else:
                row -= 1
                col += 1
        else:
            if row == block_size - 1:
                col += 1
            elif col == 0:
                row += 1
            else:
                row += 1
                col -= 1
    return unscanned_array

# Walsh-Hadamard Transform functions
def hadamard_matrix(n):
    """Generate Hadamard matrix of order n."""
    H = np.array([[1]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H

def wht_2d(block):
    """Apply Walsh-Hadamard Transform to a 2D block."""
    n = block.shape[0]
    H = hadamard_matrix(n)
    return np.dot(H, np.dot(block, H)) / n

def inverse_wht_2d(block):
    """Apply inverse Walsh-Hadamard Transform to a 2D block."""
    return wht_2d(block)

# Function to encode a grayscale image using WHT
def wht_encode_grayscale(
    img: np.ndarray, block_size: int, num_coeffs: int
) -> list[np.ndarray]:
    img_height, img_width = img.shape
    padded_height = img_height + (block_size - img_height % block_size) % block_size
    padded_width = img_width + (block_size - img_width % block_size) % block_size
    padded_img = np.zeros((padded_height, padded_width), dtype=np.uint8)
    padded_img[:img_height, :img_width] = img
    padded_img = padded_img.astype(np.float32) - 128
    blocks = [
        padded_img[i : i + block_size, j : j + block_size]
        for i in range(0, padded_height, block_size)
        for j in range(0, padded_width, block_size)
    ]
    wht_blocks = [wht_2d(block) for block in blocks]
    resized_quant_matrix = cv.resize(
        quant_matrix, (block_size, block_size), cv.INTER_CUBIC
    )
    quantized_blocks = [
        np.round(block / resized_quant_matrix).astype(np.int32) #dividing with quantisation matrix
        for block in wht_blocks
    ]
    zigzag_scanned_blocks = [perform_zigzag_scan(block) for block in quantized_blocks] #zigzag scan
    first_num_coeffs = [
        block[:num_coeffs] for block in zigzag_scanned_blocks
    ]
    return first_num_coeffs

# Function to decode a grayscale compressed image using WHT
def decode_grayscale_wht(
    blocks: list[np.ndarray], img: np.ndarray, block_size: int
) -> np.ndarray:
    img_height, img_width = img.shape
    padded_height = img_height + (block_size - img_height % block_size) % block_size
    padded_width = img_width + (block_size - img_width % block_size) % block_size
    resized_quant_matrix = cv.resize(
        quant_matrix, (block_size, block_size), cv.INTER_CUBIC #to make padding done smoothly
    )
    zigzag_unscanned_blocks = [perform_zigzag_unscan(block, block_size) for block in blocks] 
    dequantized_blocks = [
        block * resized_quant_matrix for block in zigzag_unscanned_blocks
    ]
    iwht_blocks = [inverse_wht_2d(block) for block in dequantized_blocks]
    compressed_img = np.zeros((padded_height, padded_width), dtype=np.float32)
    block_index = 0
    for i in range(0, padded_height, block_size):
        for j in range(0, padded_width, block_size):
            compressed_img[i : i + block_size, j : j + block_size] = iwht_blocks[
                block_index
            ]
            block_index += 1
    compressed_img += 128
    #compressed_img = np.clip(compressed_img, 0, 255)
    return compressed_img[:img_height, :img_width].astype(np.uint8)

# Function to encode a color image using WHT
def color_wht_encode(
    img: np.ndarray, block_size: int, num_coeffs: int
) -> tuple[
    list[np.ndarray], list[np.ndarray], list[np.ndarray]
]:
    blue_channel, green_channel, red_channel = cv.split(img)
    return (
        wht_encode_grayscale(blue_channel, block_size, num_coeffs),
        wht_encode_grayscale(green_channel, block_size, num_coeffs),
        wht_encode_grayscale(red_channel, block_size, num_coeffs),
    )

# Function to decode a color JPEG compressed image using WHT
def color_wht_decode(
    blocks: tuple[
        list[np.ndarray],
        list[np.ndarray],
        list[np.ndarray],
    ],
    img: np.ndarray,
    block_size: int,
) -> np.ndarray:
    blue_channel, green_channel, red_channel = cv.split(img)
    blue_channel = decode_grayscale_wht(blocks[0], blue_channel, block_size)
    green_channel = decode_grayscale_wht(blocks[1], green_channel, block_size)
    red_channel = decode_grayscale_wht(blocks[2], red_channel, block_size)
    return cv.merge((blue_channel, green_channel, red_channel))

# Function to encode an image (either grayscale or color) using compression with WHT
def wht_encode(
    img_path: str,
    block_size: int,
    num_coeffs: int,
    is_color: bool,
) -> (
    list[np.ndarray]
    | tuple[
        list[np.ndarray],
        list[np.ndarray],
        list[np.ndarray],
    ]
):
    if is_color:
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        return color_wht_encode(img, block_size, num_coeffs)
    else:
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        return wht_encode_grayscale(img, block_size, num_coeffs)

# Function to decode a compressed image (either grayscale or color) using WHT
def wht_decode(
    encoded_img: (
        list[np.ndarray]
        | tuple[
            list[np.ndarray],
            list[np.ndarray],
            list[np.ndarray],
        ]
    ),
    img_path: str,
    block_size: int,
    is_color: bool,
) -> np.ndarray:
    if is_color:
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        return color_wht_decode(encoded_img, img, block_size)
    else:
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        return decode_grayscale_wht(encoded_img, img, block_size)

# Function to calculate structural content between two images
def structural_content(img1: np.ndarray, img2: np.ndarray) -> float:
    return np.sum(img1**2) / np.sum(img2**2)

# Function to compute image fidelity between two images
def image_fidelity(img1: np.ndarray, img2: np.ndarray) -> float:
    return 1 - np.sum((img1 - img2)**2) / np.sum(img1**2)

# Function to analyze an image using compression with WHT and compute metrics
def analyze_image(
    img_path: str, block_size: int, num_coefficients: int, color: bool
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    list[np.ndarray]
    | tuple[
        list[np.ndarray],
        list[np.ndarray],
        list[np.ndarray],
    ],
    bool,
]:
    img: np.ndarray = None
    if color == 'y':
        img = cv.imread(img_path, cv.IMREAD_COLOR)
    else:
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    # Measure encoding time
    start_encoding = time.time()
    encoded_img = wht_encode(img_path, block_size, num_coefficients, color)
    end_encoding = time.time()
    encoding_time = end_encoding - start_encoding

    # Measure decoding time
    start_decoding = time.time()
    compressed_img = wht_decode(encoded_img, img_path, block_size, color)
    end_decoding = time.time()
    decoding_time = end_decoding - start_decoding

    # Calculate PSNR for each channel if color, otherwise for grayscale
    if color == 'y':
        psnr_values = []
        for i in range(3):
            psnr_val = compute_psnr(img[:, :, i], compressed_img[:, :, i])
            psnr_values.append(psnr_val)
        print("PSNR: ", psnr_values)
    else:
        psnr_valu = compute_psnr(img, compressed_img)
        print("PSNR: ", psnr_valu)

    # Calculate SSIM for each channel if color, otherwise for grayscale
    if color == 'y':
        ssim_values = []
        for i in range(3):
            ssim_val = ssim(img[:, :, i], compressed_img[:, :, i], win_size=7)
            ssim_values.append(ssim_val)
        print("SSIM: ", ssim_values)
    else:
        ssim_valu = ssim(img, compressed_img)
        print("SSIM: ", ssim_valu)

    # Calculate additional metrics
    sc_value = structural_content(img, compressed_img)
    if_value = image_fidelity(img, compressed_img)
    print("Structural Content: ", sc_value)
    print("Image Fidelity: ", if_value)

    # Calculate the compression ratio
    n2 = total_elements_in_blocks(encoded_img, color)
    if n2 == 0:
        # In this case, the compression ratio is very high
        # But, we set it to 0 to avoid division by 0 so that our analysis becomes easier
        compression_ratio = 0
    else:
        compression_ratio = img.size / n2

    # Save the decompressed image
    if color == 'y':
        cv.imwrite("wht//compressed_image.png", compressed_img)
    else:
        cv.imwrite("wht//compressed_image.png", compressed_img)

    # Print the results
    print("Compression Ratio: ", compression_ratio)
    print("Encoding Time: ", encoding_time, "seconds")
    print("Decoding Time: ", decoding_time, "seconds")

    return (img, compressed_img, psnr_valu, compression_ratio, encoded_img, color)

if __name__ == "__main__":
    # Input parameters from the user
    img_path = input("Enter the path to the image: ")
    block_size = int(input("Enter the block size (even): "))
    num_coefficients = int(input("Enter the number of coefficients passed: "))
    color_input = input("Is the image color (y/n): ").strip().lower()
    color = color_input == 'y'

    # Analyze the image using the provided parameters
    analyze_image(img_path, block_size, num_coefficients, color)
