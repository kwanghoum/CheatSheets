
i, a, o, shift + i, shift + a, cw : 명령모드 -> 입력모드
esc : 입력모드 -> 명령모드

h,j,k,l
shift + $, shift + ^

shift + v
y : 복사
p : 붙여넣기
d : 지우기
yy : 해당 라인 복사
dd : 해당 라인 지우기
4yy : 두줄 복사
2dd : 두줄 지우기

ctrl + f : 지금 보이는 화면의 다음 라인
ctrl + b : 위와 반대

shift + j

D : 현재 커서가 있는 곳부터 해당라인의 뒷부분 삭제

shift + insert : 다른 곳에서 복사한거 붙여넣을 때 (입력모드상태에서)

여러라인 선택 + = : 줄맞춤
중괄호 위 커서 + = + % : 중괄호 안 블록 줄맞춤

* : 커서 위의 단어와 같은 단어 찾아줌 (n : 다음 위치, N : 이전 위치)

u : undo
ctrl + r : redo

x : 한 문자 지움
6x : 여섯 문자 지움

### Line Mode
/fdsa : 'fdsa'문자열 찾아줌 (n, N)

:70 : 70번째 줄로 이동
:1
:$ : 제일 마지막 줄

:set nonumber
:set number

:%s/찾을문자열/바꿀문자열

:w
:q
:wq

:q! : 저장안하고 나가기


출처 : <https://www.youtube.com/watch?v=GWo_MxMlJJ4&ab_channel=%EC%8B%9C%EA%B3%A8%EC%82%AC%EB%8A%94%EA%B0%9C%EB%B0%9C%EC%9E%90>