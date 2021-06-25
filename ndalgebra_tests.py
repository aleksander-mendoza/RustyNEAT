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

x = nd.array([[0, 1, 0],
              [1, 0, 0],
              [0, 0, 1]], context)
y = nd.array([[0, 1, 2],
              [3, 4, 5],
              [6, 7, 8]], context)
z = x + y
assert str(z) == "[[0, 2, 2], [4, 4, 5], [6, 7, 9]]"
z += x
assert str(z) == "[[0, 3, 2], [5, 4, 5], [6, 7, 10]]"
z = x - y
assert str(z) == "[[0, 0, -2], [-2, -4, -5], [-6, -7, -7]]"
z -= x
assert str(z) == "[[0, -1, -2], [-3, -4, -5], [-6, -7, -8]]"
z = (x + x) * y
assert str(z) == "[[0, 2, 0], [6, 0, 0], [0, 0, 16]]"
z[:, :] = y
assert str(z) == "[[0, 1, 2], [3, 4, 5], [6, 7, 8]]"
z *= x
assert str(z) == "[[0, 1, 0], [3, 0, 0], [0, 0, 8]]"
z = x+1
assert str(z) == "[[1, 2, 1], [2, 1, 1], [1, 1, 2]]"
z = x*2
assert str(z) == "[[0, 2, 0], [2, 0, 0], [0, 0, 2]]"
z = x-10
assert str(z) == "[[-10, -9, -10], [-9, -10, -10], [-10, -10, -9]]"
z = x @ y
assert str(z) == "[[3, 4, 5], [0, 1, 2], [6, 7, 8]]"
z = y @ x
assert str(z) == "[[1, 0, 2], [4, 3, 5], [7, 6, 8]]"
z = y / (x+1)
assert str(z) == "[[0, 0, 2], [1, 4, 5], [6, 7, 4]]"
y_f32 = y.astype(nd.float32)
assert y_f32.dtype == nd.float32
x_f32 = x.astype(nd.float32)
assert x_f32.dtype == nd.float32
z = y_f32 / (x_f32+1)
assert str(z) == "[[0, 0.5, 2], [1.5, 4, 5], [6, 7, 4]]"
assert str(x) == "[[0, 1, 0], [1, 0, 0], [0, 0, 1]]"
assert str(y) == "[[0, 1, 2], [3, 4, 5], [6, 7, 8]]"

