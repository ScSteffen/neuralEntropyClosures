def f_1(a: str) -> str:
    return a


def f_1(a: int) -> int:
    return a - 1


a: str = "test"
b: int = 2

print(f_1(a))
print(f_1(b))
