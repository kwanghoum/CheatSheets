reading data
```python
n = int(input())
n, m, k = map(int, input().split())
data = list(map(int, input().split()))

import sys
data = sys.stdin.readline().rstrip()

```

lambda expression (:function without a name)
(use when theres no need to create a function with name
```python
#lambda args: expression
f1 = lambda n1, n2: n1 + n2
f(3,4) # >>>7
```
map function
```python
numbers = list(range(1, 11))
map = list(map(lambda n: n**2, numbers))  #argument: (function, iterable object)
```
filter function
```python
numbers = list(range(1, 11))
filter = list(filter(lambda n: n % 2, numbers)) #argument: (function that return True or False, iterable object)
```
list comprehension
```python
numbers = list(range(1, 11))
map_&_filter = list(map(lambda n: n**2, filter(lambda n: n % 2, numbers)))
list_comprehension = [n**2 for n in numbers if n % 2]
```

binary search
```python
from bisect import bisect_left, bisect_right

def count_by_range(list, left_value, right_value):
    right_index = bisect_right(list, right_value) #정렬된 순서를 유지하면서 리스트 list에 데이터 right_value를 삽입할 가장 오른쪽 인덱스를 찾는 메소드
    left_index = bisect_left(list, left_value) #정렬된 순서를 유지하면서 리스트 list에 데이터 left_value를 삽입할 가장 왼쪽 인덱스를 찾는 메소드
    return right_index - left_index
```
range(start, stop[, step])
```python
for i in range(n, -1, -1):
    print(i)
... 
n
9
8
7
6
5
4
3
2
1
0

```
