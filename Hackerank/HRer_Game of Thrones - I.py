s = "cdefghmnopqrstuvw"
Lower = "abcdefghijklmnopqrstuvwxyz"
Button = True
for ele in Lower:
    print(s.count(ele))
    if s.count(ele)%2 ==0:
        pass
    elif Button:
        Button = False
        continue
    else:
        print("NO")
        break
    print("YES")