def longestSubstring(searchString, lookAheadString):
    max_length = 0
    min_distance_index = float('inf')
    for i in range(len(searchString) - len(lookAheadString) + 1):
        length = 0
        distance = 0
        while length < len(lookAheadString) and searchString[i + length] == lookAheadString[length]:
            length += 1
        if length > max_length:
            max_length = length
            min_distance_index = i
        elif length == max_length:
            for j in range(length):
                if searchString[i + j] != lookAheadString[j]:
                    distance = j
                    break
            if distance < min_distance_index:
                min_distance_index = i

    return max_length, min_distance_index


def encode_lz77(data, searchBufferSize, lookAheadBufferSize):
    encodedNums = []
    encodedLengths = []
    encodedLetters = []
    i = 0
    while i < len(data):
        if i < 2:
            if i == 0:
                encodedNums.append(0)
                encodedLengths.append(0)
                encodedLetters.append(data[i])
                i=i+1
                continue
            else:
                if (i > 0 and data[i] != data[0]):
                    encodedNums.append(0)
                    encodedLengths.append(0)
                    encodedLetters.append(data[i])
                    i=i+1
                    continue
                else:
                    encodedNums.append(1)
                    encodedLengths.append(1)
                    encodedLetters.append(data[i+1])
                    i = i + 2
                    continue
            
        else:
            lookAheadString = data[i:i+lookAheadBufferSize]
            searchBufferindex = 0
            if (i < searchBufferSize):
                searchBufferindex = i
            else:
                searchBufferindex = searchBufferSize
            searchString = data[i - searchBufferindex:i]
            result = longestSubstring(searchString + lookAheadString, lookAheadString)
            encodedLetter = ''
            if (result[0] == len(lookAheadString)):
                if (i + result[0] == len(data)):
                    encodedLetter = ''
                else:
                    encodedLetter = data[i+lookAheadBufferSize]
            else:
                encodedLetter = lookAheadString[result[0]]
            if (result[0] == 0):
                encodedNums.append(0)
                encodedLengths.append(0)
                encodedLetters.append(encodedLetter)
            else:
                encodedNums.append(searchBufferindex - result[1])
                encodedLengths.append(result[0])
                encodedLetters.append(encodedLetter)
            i = i + result[0] + 1
    return encodedNums, encodedLengths, encodedLetters


def decode_lz77(encodedNums, encodedLengths, encodedLetters):
    i = 0
    decodedString = []

    while i < len(encodedNums):
        if (i == 1 and encodedLetters[i] == encodedLetters[0]):
            decodedString.append(encodedLetters[i])
            i = i+1
        else:
            if (encodedNums[i] == 0):
                decodedString.append(encodedLetters[i])
            else:
                currentSize = len(decodedString)
                for j in range(0, encodedLengths[i]):
                    decodedString.append(
                        decodedString[currentSize-encodedNums[i]+j])
                decodedString.append(encodedLetters[i])
            i = i+1
    return decodedString
number=input ('If you have compress "1" and decompress "0" : ' )



if number=='1' :
    file = input("Enter the filename you want to compress: ")
    with open(file, 'r') as f:
        stringToEncode = f.read()
    restrictions = int(input("Do you want restrictions on searchwindow or look ahead window enter '1' if yes '0' if no: "))
    if (restrictions == 0):
        [encodedNums, encodedLengths, encodedLetters] = encode_lz77(stringToEncode,len(stringToEncode),len(stringToEncode))
    else:
        searchBufferSize = int(input("Enter the Search Buffer Size: "))
        lookAheadBufferSize = int(input("Enter the look ahead buffer Size: "))
        [encodedNums, encodedLengths, encodedLetters] = encode_lz77(stringToEncode, searchBufferSize, lookAheadBufferSize)
    a = [encodedNums, encodedLengths, encodedLetters]
   
    output = open("encoded.txt","w+")
    size = len(a[0])
    for i in range(size):
        word = "<" + str(encodedNums[i]) + ":" + str(encodedLengths[i]) + ":" + encodedLetters[i] + ">"
        output.write(word)
    print("Compressed file generated as encoded.txt")

else:
    encodedNums_d = []
    encodedLengths_d = []
    encodedLetters_d = []
    file = input("Enter the filename you want to decompress:")
    f = open(file, "r").readline()
    newList = f.split(">")
    for word in newList:
        if word == '':
            break
        newword = word[1:].split(":")
        encodedNums_d.append(int(newword[0]))
        encodedLengths_d.append(int(newword[1]))
        encodedLetters_d.append(newword[2])

    decodedString = decode_lz77(encodedNums_d, encodedLengths_d, encodedLetters_d)
    output = open("decoded.txt","w+")
    output.write("".join(decodedString))
    print("Decoded file generated as decoded.txt")