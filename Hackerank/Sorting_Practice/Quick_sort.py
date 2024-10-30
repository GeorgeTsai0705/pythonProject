import random
import time

original = [random.randint(1, 300) for x in range(15)]
print(original)


def swap(data, i, j):
    temp = data[i]
    data[i] = data[j]
    data[j] = temp


def quick_sort(data, left, right):
    if left < right:
        i = left
        j = right + 1
        while True:

            while i + 1 < len(data) and data[i + 1] < data[left]:
                i += 1
            j -= 1
            while j > -1 and data[j] > data[left]:
                pass
            if i >= j:
                break
            print(data, i, j)
            swap(data, i, j)

        swap(data, left, j)
        quick_sort(data, left, j - 1)
        quick_sort(data, j + 1, right)
    return data


print(quick_sort(original, 0, len(original) - 1))
