Nums = [0 for i in range(100)]
Count = 0
f = open('input07.txt','r')

for line in f.readlines():
    M, D =map(int, line.split(' '))

    if int(M) <= (D//10) or M <= (D%10):
        continue
    Nums[M*(D//10) + (D%10)] += 1
for i in Nums:
    Count += i*(i-1)*0.5
print(int(Count))
f.close()