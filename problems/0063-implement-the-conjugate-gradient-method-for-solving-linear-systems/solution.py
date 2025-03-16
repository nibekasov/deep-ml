import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
	"""
	Solve the system Ax = b using the Conjugate Gradient method.

	:param A: Symmetric positive-definite matrix
	:param b: Right-hand side vector
	:param n: Maximum number of iterations
	:param x0: Initial guess for solution (default is zero vector)
	:param tol: Convergence tolerance
	:return: Solution vector x
	"""
	if x0 is None:
		x = np.zeros_like(b)
	else:
		x = x0.copy()

	r = b - np.dot(A, x)
	p = r.copy()
	rs_old = np.dot(r,r)

	for i in range(n):
		Ap = np.dot(A, p)
		alpha = rs_old / np.dot(p, Ap)
		x = x + alpha * p
		r = r - alpha * Ap
		rs_new = np.dot(r, r)

		if np.sqrt(rs_new) < tol:
			# print(f"Converged after {i+1} iterations.")
			break

		beta = rs_new / rs_old
		p = r + beta * p
		rs_old = rs_new

	return x

	x = np.zeros_like(b)
