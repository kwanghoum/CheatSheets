VOICE FISHING
reading data
```python
n = int(input())
n, m, k = map(int, input().split())
data = list(map(int, input().split()))

import sys
data = sys.stdin.readline().rstrip()

```

```python
x.isalpha() #알파벳인지 숫자인지 확인
```
리스트 -> 스트링
```python
"".join(list)
```
```python
list.pop(0)
list.append(something)
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
sort
```python
data = [('Kim', 33), ('Lee', 26), ('Um', 29)]
def age(t):
    return t[1]
data.sort(key = age) #나이 어린 순으로 정렬
data.sort(key = lambda t : t[1], reverse = True) #나이 많은 순으로 정렬
data.sort(key = lambda t : t[0]) #이름의 알파벳순으로 정렬 (알파벳은 순서가 뒤로 갈수록 값이 큼)
#sorted() 함수는 원본을 유지. 튜플은 sort()함수 못쓰고, sorted()함수 쓰면 결과는 리스트.
org = (3, 1, 2)
cpy = tuple(sorted(org))
```
enumerate (enumerate함수는 리스트와 같은 'iterable객체'를 argument로 받아서 iterable객체를 새로 생성하여 짝지어 반환함)
```python
for i, name in enumerate(['body', 'foo', 'bar'], 10): # 번호를 10부터 매김
    print(i, name)
...
10 body
11 foo
12 bar

#이름들 있는 리스트를 이름 알파벳 순으로 정렬하고, 1부터 번호매기는 딕셔너리 만들기
names = ['Um', 'Lee', 'Kim']
dnames = {k : v for k, v in enumerate(sorted(names), 1}
...
{1: 'Kim', 2: 'Lee', 3: 'Um'}
```
표현식기반문자열조합 & 메소드 기반 문자열 조합
```python
```

binary search
```python
from bisect import bisect_left, bisect_right

def count_by_range(list, left_value, right_value):
    right_index = bisect_right(list, right_value) # 정렬된 순서를 유지하면서 리스트 list에
                                                    데이터 right_value를 삽입할 가장 오른쪽 인덱스를 찾는 메소드
    left_index = bisect_left(list, left_value) # 정렬된 순서를 유지하면서 리스트 list에 
                                                    데이터 left_value를 삽입할 가장 왼쪽 인덱스를 찾는 메소드
    return right_index - left_index
```
파라메트릭 서치 문제 유형은 이진탐색을 재귀적말고 반복문을 용해 구현하면 더 간결하게 문제풀 수 있음. (둘 다 외워두자)
```python
''' Binary Search (Iterative Method) '''
def binary_search(array, target, start, end):
    while start <= end:
        mid = (start + end) // 2
        # If the target is found, return the mid index.
        if array[mid] == target:
            return mid
        # If the value of the mid index is greater than the target, search the left part.
        elif array[mid] > target:
            end = mid - 1
        # If the value of the mid index is smaller than the target, search the right part.
        else:
            start = mid + 1
    return None
    
''' Binary Search (recursive method) '''
def binary_search(array, target, start, end):
    if start > end:
        return None
    mid = (start + end) // 2
    if array[mid] == target:
        return mid
    elif array[mid] > target:
        return binary_search(array, target, start, mid - 1)
    else:
        return binary_search(array, target, mid + 1, end)

n = 10
target = 13
array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

result = binary_search(array, target, 0, n - 1)
if result == None:
    print(None)
else:
    print(result + 1)
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
disjoint set(서로소 집합) & Kruskal's algorithm(크루스칼 알고리즘)
```python
#두 노드가 있을 때 각 노드의 루트 노드가 동일하다면 같은 집합(그래프/트리)에 포함된 것.
#방향성이 없는 경우라도 일반적으로 번호가 작은 노드가 큰 노드의 루트 노드가 되도록 방향을 잡음.

#한 노드의 루트 노드를 찾는 함수
def find_parent(parent, x):
    if parent[x] !=x:
        parent[x] = find_parent(parent, parent[x])
       return parent[x]
#각 노드가 속한 그래프(각각 서로소 집합 관계)를 한 그래프로 합치는 함수
def union_parent(parent, a, b):
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a < b:
        parent[b] = a
    else:
        parent[a] = b
parent = [0] * (n + 1)   # n: 노드의 개수 / parent는 각 노드들의 루트(부모) 노드를 담는 리스트 이다.

for i in range(1, n + 1):  # 루트 노드 테이블상에서, 루트 노드들을 자기 자신으로 초기화
    parent[i] = i
    
#크루스칼 알고리즘이란 여러 노드들(과 간선들)이 주어져있을 때 최소의 비용으로 모든 노드를 연결하는 방법(사용할 간선들)을 찾는 알고리즘.
#모든 노드가 연결된 그래프 = 신장 트리
#다른 말로, 크루스칼 알고리즘이란 최소의 비용으로 신장트리를 만드는 알고리즘.
#여기서 비용이란 일반적으로 간선의 크기를 의미.

edges = []
for _ range(m):  # m : 간선의 개수
    a, b, cost = map(int, input().split()) # 노드, 노드, 간선크기
    edges.append((cost, a, b))
edges.sort()

for edge in edges:
    cost, a, b = edge
    if find_parent(parent, a) != find_parent(parent, b):
        union_parent(parent, a, b)
#위와 같이 [(간선크기, 노드, 노드), ~ ,(간선크기, 노드, 노드)]로 구성된 간선의 정보를 간선크기로 정렬해서
#차례대로 노드를 연결하는 방식으로 신장트리를 만듬(=크루스칼 알고리즘)
```
