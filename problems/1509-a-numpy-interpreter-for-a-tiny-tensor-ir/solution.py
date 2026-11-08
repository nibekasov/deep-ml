import numpy as np


def evaluate(node, bufs, cache=None):
    if cache is None:
        cache = {}

    # Этот узел уже вычисляли
    if node in cache:
        return cache[node]

    op = node[0]

    # Вычисление дочернего узла с общим кешем
    def ev(child):
        return evaluate(child, bufs, cache)

    if op == "buf":
        result = bufs[node[1]]

    elif op == "const":
        result = np.asarray(node[1], dtype=np.float32)

    elif op in ("add", "mul", "max"):
        a = ev(node[1])
        b = ev(node[2])

        if op == "add":
            result = a + b
        elif op == "mul":
            result = a * b
        else:
            result = np.maximum(a, b)

    elif op in ("neg", "exp", "recip"):
        a = ev(node[1])

        if op == "neg":
            result = -a
        elif op == "exp":
            result = np.exp(a)
        else:
            result = np.reciprocal(a)

    elif op == "reshape":
        result = ev(node[1]).reshape(node[2])

    elif op == "expand":
        result = np.broadcast_to(ev(node[1]), node[2])

    elif op == "permute":
        result = np.transpose(ev(node[1]), axes=node[2])

    elif op == "flip":
        result = np.flip(ev(node[1]), axis=node[2])

    elif op == "pad":
        result = np.pad(
            ev(node[1]),
            pad_width=node[2],
            mode="constant",
            constant_values=0,
        )

    elif op == "shrink":
        slices = tuple(
            slice(start, end)
            for start, end in node[2]
        )
        result = ev(node[1])[slices]

    elif op == "sum":
        result = np.sum(
            ev(node[1]),
            axis=node[2],
            keepdims=True,
            dtype=np.float32,
        )

    elif op == "amax":
        result = np.max(
            ev(node[1]),
            axis=node[2],
            keepdims=True,
        )

    else:
        raise ValueError(f"Unknown operation: {op}")

    # Приведение без копирования, если dtype уже float32
    result = np.asarray(result, dtype=np.float32)

    cache[node] = result
    return result


def eval_count(node, bufs):
    cache = {}
    evaluate(node, bufs, cache)
    return len(cache)
