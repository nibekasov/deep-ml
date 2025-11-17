import numpy as np

def mutual_information(joint_prob: list[list[float]]) -> float:
	"""
	Compute the mutual information between two random variables.
	
	Args:
		joint_prob: 2D joint probability distribution P(X,Y)
	
	Returns:
		Mutual information I(X;Y)
	"""
	joint = np.array(joint_prob, dtype=float)

    # Marginals: P(X) and P(Y)
    px = np.sum(joint, axis=1)  # sum over columns → P(X)
    py = np.sum(joint, axis=0)  # sum over rows    → P(Y)

    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            p_xy = joint[i, j]
            if p_xy > 0 and px[i] > 0 and py[j] > 0:
                mi += p_xy * np.log(p_xy / (px[i] * py[j]))

    return float(mi)