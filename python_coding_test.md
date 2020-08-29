
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
