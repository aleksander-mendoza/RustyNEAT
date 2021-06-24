import rusty_neat
from rusty_neat import ndalgebra as nd

context = rusty_neat.make_gpu_context()  # First create OpenCL context

x = nd.array([1, 2, 3, 4], context, dtype=nd.f32)
assert str(x) == "[1, 2, 3, 4]"
a, b, c, d = x
assert (a, b, c, d) == (1, 2, 3, 4)
assert x[0] == 1
assert x[1] == 2
assert x[2] == 3
assert x[3] == 4

x = nd.array([[1], [2], [3], [4]], context, dtype=nd.float32)
assert str(x) == "[[1], [2], [3], [4]]"
a, b, c, d = x
assert (str(a), str(b), str(c), str(d)) == ("[1]", "[2]", "[3]", "[4]")
a, b, c, d = a.item(), b.item(), c.item(), d.item()
assert (a, b, c, d) == (1, 2, 3, 4)
assert str(x[0]) == "[1]"

x[:, :] = 6
assert str(x) == "[[6], [6], [6], [6]]"

x = x.reshape(2, 2)
assert str(x) == "[[6, 6], [6, 6]]"
x[0, :] = 2
assert str(x) == "[[2, 2], [6, 6]]"

y = nd.array([1, 2], context)
x[:, 0] = y
assert str(x) == "[[1, 2], [2, 6]]"
x[:, 1] = [14, 32]
assert str(x) == "[[1, 14], [2, 32]]"
x[:, 1] = -3
assert str(x) == "[[1, -3], [2, -3]]"
