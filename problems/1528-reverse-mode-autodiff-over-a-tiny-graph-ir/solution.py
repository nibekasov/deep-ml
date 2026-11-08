import numpy as np


def children(node):
    op = node[0]

    if op in ("buf", "const"):
        return ()

    if op in ("add", "mul", "max"):
        return (node[1], node[2])

    return (node[1],)


def topological_order(root):
    order = []
    visited = set()
    stack = [(root, False)]

    while stack:
        node, ready = stack.pop()

        if node in visited:
            continue

        if ready:
            visited.add(node)
            order.append(node)
        else:
            stack.append((node, True))

            for child in reversed(children(node)):
                stack.append((child, False))

    return order


def _evaluate(node, bufs, cache, dtype):
    if node in cache:
        return cache[node]

    for current in topological_order(node):
        if current in cache:
            continue

        op = current[0]

        if op == "buf":
            result = np.asarray(bufs[current[1]], dtype=dtype)

        elif op == "const":
            result = np.asarray(current[1], dtype=dtype)

        else:
            a = cache[current[1]]

            if op == "add":
                result = a + cache[current[2]]

            elif op == "mul":
                result = a * cache[current[2]]

            elif op == "max":
                result = np.maximum(a, cache[current[2]])

            elif op == "neg":
                result = -a

            elif op == "exp":
                result = np.exp(a)

            elif op == "recip":
                result = np.reciprocal(a)

            elif op == "reshape":
                result = a.reshape(current[2])

            elif op == "expand":
                result = np.broadcast_to(a, current[2])

            elif op == "permute":
                result = np.transpose(a, axes=current[2])

            elif op == "sum":
                result = np.sum(
                    a,
                    axis=current[2],
                    keepdims=True,
                    dtype=dtype,
                )

            else:
                raise NotImplementedError(
                    f"Unknown operation: {op}"
                )

        cache[current] = np.asarray(result, dtype=dtype)

    return cache[node]


def forward(node, bufs, cache):
    return _evaluate(node, bufs, cache, np.float32)


def backward(loss, bufs, wrt):
    cache = {}
    loss_value = forward(loss, bufs, cache)

    if loss_value.size != 1:
        raise ValueError("loss must contain exactly one element")

    order = topological_order(loss)
    grads = {loss: np.ones_like(loss_value)}

    def accumulate(node, contribution):
        contribution = np.asarray(
            contribution, dtype=np.float32
        )

        if node in grads:
            grads[node] = grads[node] + contribution
        else:
            grads[node] = contribution

    for node in reversed(order):
        g = grads[node]
        op = node[0]

        if op in ("buf", "const"):
            continue

        a = node[1]

        if op == "sum":
            accumulate(
                a,
                np.broadcast_to(g, cache[a].shape),
            )

        elif op == "mul":
            b = node[2]
            accumulate(a, g * cache[b])
            accumulate(b, g * cache[a])

        elif op == "add":
            accumulate(a, g)
            accumulate(node[2], g)

        elif op == "neg":
            accumulate(a, -g)

        elif op == "exp":
            accumulate(a, g * cache[node])

        elif op == "recip":
            out = cache[node]
            accumulate(a, -g * out * out)

        elif op == "max":
            b = node[2]
            choose_a = cache[a] >= cache[b]

            accumulate(a, np.where(choose_a, g, 0.0))
            accumulate(b, np.where(choose_a, 0.0, g))

        elif op == "reshape":
            accumulate(a, g.reshape(cache[a].shape))

        elif op == "permute":
            inverse_perm = tuple(np.argsort(node[2]))
            accumulate(
                a,
                np.transpose(g, axes=inverse_perm),
            )

        elif op == "expand":
            input_shape = cache[a].shape

            padded_shape = (
                (1,) * (g.ndim - len(input_shape))
                + input_shape
            )

            axes = tuple(
                i
                for i, (before, after) in enumerate(
                    zip(padded_shape, g.shape)
                )
                if before == 1 and after != 1
            )

            contribution = np.sum(
                g,
                axis=axes,
                keepdims=True,
                dtype=np.float32,
            )

            accumulate(
                a, contribution.reshape(input_shape)
            )

        else:
            raise NotImplementedError(
                f"No gradient rule for {op}"
            )

    return {
        name: grads.get(
            ("buf", name),
            np.zeros_like(bufs[name], dtype=np.float32),
        )
        for name in wrt
    }


def numeric_gradient(loss, bufs, name, eps=1e-3):
    work_bufs = {
        key: np.array(value, dtype=np.float64, copy=True)
        for key, value in bufs.items()
    }

    x = work_bufs[name]
    grad = np.zeros_like(x, dtype=np.float64)

    for index in np.ndindex(x.shape):
        original = x[index].item()

        try:
            x[index] = original + eps
            plus = _evaluate(
                loss, work_bufs, {}, np.float64
            ).item()

            x[index] = original - eps
            minus = _evaluate(
                loss, work_bufs, {}, np.float64
            ).item()
        finally:
            x[index] = original

        grad[index] = (plus - minus) / (2 * eps)

    return grad

def _topo(node):
    return topological_order(node)
