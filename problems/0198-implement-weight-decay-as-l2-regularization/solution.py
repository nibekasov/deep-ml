def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
	"""
	Apply weight decay (L2 regularization) to parameters.
	
	Args:
		parameters: List of parameter arrays
		gradients: List of gradient arrays
		lr: Learning rate
		weight_decay: Weight decay factor
		apply_to_all: Boolean list indicating which parameter groups get weight decay
	
	Returns:
		Updated parameters
	"""
	updated_params: List[List[float]] = []

    for group_params, group_grads, apply_decay in zip(parameters, gradients, apply_to_all):
        new_group = []
        for w, g in zip(group_params, group_grads):
            if apply_decay:
                # w_new = w - lr * g - lr * weight_decay * w
                new_w = w - lr * g - lr * weight_decay * w
            else:
                # No weight decay: plain SGD
                new_w = w - lr * g
            n