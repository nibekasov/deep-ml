import math

def softplus(x: float) -> float:
	"""
	Compute the softplus activation function.

	Args:
		x: Input value

	Returns:
		The softplus value: log(1 + e^x)
	"""
	if x > 0:
        res = x + math.log1p(math.exp(-x))  # math.log1p(y) == log(1+y), точнее для малых y
    else:
        res = math.log1p(math.exp(x))
    return round(res, 4)