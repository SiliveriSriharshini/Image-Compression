def encoding(s1):
    print("Encoding")
    table = {chr(i): i for i in range(256)}
    p = ""
    p += s1[0]
    code = 256
    output_code = []
    print("String\tOutput_Code\tAddition")
    
    for i in range(len(s1)):
        if i != len(s1) - 1:
            c = s1[i + 1]
        else:
            c = ''
        
        if p + c in table:
            p = p + c
        else:
            print(f"{p}\t{table[p]}\t\t{p + c}\t{code}")
            output_code.append(table[p])
            table[p + c] = code
            code += 1
            p = c
            
    print(f"{p}\t{table[p]}")
    output_code.append(table[p])
    return output_code

def decoding(op):
    print("\nDecoding")
    table = {i: chr(i) for i in range(256)}
    old = op[0]
    s = table[old]
    c = ""
    c += s[0]
    print(s, end="")
    count = 256
    
    for i in range(len(op) - 1):
        n = op[i + 1]
        if n not in table:
            s = table[old]
            s = s + c
        else:
            s = table[n]
        
        print(s, end="")
        c = ""
        c += s[0]
        table[count] = table[old] + c
        count += 1
        old = n

if __name__ == "__main__":
    s = "WYS*WYGWYS*WYSWYSG"
    output_code = encoding(s)
    print("Output Codes are: ", end="")
    for code in output_code:
        print(code, end=" ")
    print()
    decoding(output_code)
