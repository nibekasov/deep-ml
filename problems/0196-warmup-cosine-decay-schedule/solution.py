import math


def warmup_cosine_schedule(T: int, W: int, lr_max: float, lr_min: float) -> list[float]:
	"""
	Compute learning rate schedule with linear warmup and cosine decay.
	
	Args:
		T: Total number of training steps
		W: Number of warmup steps
		lr_max: Maximum learning rate (reached after warmup)
		lr_min: Minimum learning rate (reached at end of training)
	
	Returns:
		List of learning rates for each step
	"""
	lrs = []

    for t in range(T):
        if t < W:
            # Linear warmup from 0 → lr_max
            lr = (t / W) * lr_max if W > 0 else lr_max
        else:
            # Cosine decay from lr_max → lr_min over (T - W) steps
            progress = (t - W) / (T - W)
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
        
        lrs.append(round(lr, 4))

    return lrs