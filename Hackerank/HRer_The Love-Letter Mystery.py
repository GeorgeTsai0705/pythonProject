s = "abcd"
Count = 0
while len(s) >1:
    Count += abs(ord(s[-1]) - ord(s[0]))
    s = s[1:-1]
print(Count)