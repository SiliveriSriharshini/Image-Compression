def run_length_encoding(s):
    encoded_list = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            encoded_list.append((s[i - 1], count))
            count = 1
    encoded_list.append((s[-1], count))
    return encoded_list

# Test the function
string_to_encode = "AAAABBBCCDAA"
encoded_result = run_length_encoding(string_to_encode)
print(encoded_result)
