import random
import time

original = [random.randint(1, 300) for x in range(15)]
print(original)


def swap(data, i, j):
    temp = data[i]
    data[i] = data[j]
    data[j] = temp


def shaker_sort(data):
    left = 0
    right = len(data) - 1
    shift = 0

    while left < right:
        for i in range(left, right):
            if data[i] > data[i + 1]:
                swap(data, i, i + 1)
                shift = i
        right = shift
        for i in range(right, left, -1):
            if data[i] < data[i - 1]:
                swap(data, i, i - 1)
                shift = i
        left = shift
    return data


print(shaker_sort(original))
