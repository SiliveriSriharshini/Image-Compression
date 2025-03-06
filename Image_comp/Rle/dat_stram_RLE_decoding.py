def rle_decode(encoded_sequence):
    """
    Decodes a run-length encoded sequence and returns it as a string.
    
    Parameters:
    encoded_sequence (list of tuples): The encoded sequence where each tuple is (count, value).
    
    Returns:
    str: The decoded sequence as a string.
    """
    decoded_string = ''
    
    for count, value in encoded_sequence:
        decoded_string += value * count
    
    return decoded_string

def get_user_input():
    """
    Gets the RLE encoded sequence from the user as a list of tuples.
    
    Returns:
    list of tuples: The encoded sequence where each tuple is (count, value).
    """
    user_input = input("Enter the RLE encoded sequence as a list of tuples (e.g., [(3, 'a'), (1, 'b'), (2, 'c')]): ")
    try:
        encoded_sequence = eval(user_input)
        # Validate that the input is a list of tuples with correct structure
        if isinstance(encoded_sequence, list) and all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int) for item in encoded_sequence):
            return encoded_sequence
        else:
            print("Invalid format. Please ensure the input is a list of tuples (e.g., [(3, 'a'), (1, 'b'), (2, 'c')]).")
            return get_user_input()
    except:
        print("Invalid format. Please ensure the input is a list of tuples (e.g., [(3, 'a'), (1, 'b'), (2, 'c')]).")
        return get_user_input()

# Get encoded sequence from user input
encoded_sequence = get_user_input()

# Decode the sequence
decoded_string = rle_decode(encoded_sequence)

# Print the decoded string
print("Decoded string:", decoded_string)

