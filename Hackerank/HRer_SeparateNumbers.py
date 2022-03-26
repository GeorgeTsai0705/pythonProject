string = '99100'

def separateNumbers(s):
    L = len(s)
    if s[0] =='0':
        if CHECK_NUM(s,L,1):
            print('YES', 0)
            return 0
        else:
            print('NO')
            return 0

    for i in range(1, L//2+1):
        if not CHECK_NUM(s,L,i):
            continue
        else:
            print('YES',s[0:i])
            return 0
    print('NO')
    return 0


def CHECK_NUM(s,L,i):
    Position = 0
    while Position + i < L:
        A = int(s[Position : Position+i])
        print(A,Position)
        if len( str(A+1) ) > i:
            B = int(s[Position + i:Position + 2*i+1])
            Position += i
            i += 1
        else:
            B = int(s[Position + i:Position + 2 * i ])
            Position += i

        if(A + 1 != B):
            return False
    return True

separateNumbers(string)