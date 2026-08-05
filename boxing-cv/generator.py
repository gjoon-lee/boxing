def gen():
    try:
        print("A")
        yield 1
        print("B")
        yield 2
        print("C")
    finally:
        print("cleanup")

for x in gen():
    print("got", x)
    break

print("after loop")