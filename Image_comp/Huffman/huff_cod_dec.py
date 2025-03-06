import heapq
from collections import defaultdict, Counter

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
    heapq.heapify(heap)
    
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def huffman_encoding(data):
    if not data:
        return "", {}
    
    frequency = calculate_frequency(data)
    huffman_tree = build_huffman_tree(frequency)
    
    huff_dict = {symbol: code for symbol, code in huffman_tree}
    encoded_data = ''.join(huff_dict[symbol] for symbol in data)
    
    return encoded_data, huff_dict

def huffman_decoding(encoded_data, huff_dict):
    if not encoded_data or not huff_dict:
        return ""
    
    reverse_huff_dict = {v: k for k, v in huff_dict.items()}
    current_code = ""
    decoded_data = []
    
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_huff_dict:
            decoded_data.append(reverse_huff_dict[current_code])
            current_code = ""
    
    return ''.join(decoded_data)

if __name__ == "__main__":
    data = "this is an example for huffman encoding"
    
    encoded_data, huff_dict = huffman_encoding(data)
    print(f"Encoded data: {encoded_data}")
    print(f"Huffman Dictionary: {huff_dict}")
    
    decoded_data = huffman_decoding(encoded_data, huff_dict)
    print(f"Decoded data: {decoded_data}")
