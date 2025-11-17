import math

def chi_square_probability(x, k):
    """
    Calculate the probability density of x in a Chi-square distribution
    with k degrees of freedom.
    """
    if x < 0:
        return 0.0  # PDF is zero for negative x
    
    # compute PDF
    coeff = 1 / ( (2 ** (k / 2)) * math.gamma(k / 2) )
    power = x ** (k/2 - 1)
    exp_term = math.exp(-x / 2)
    
    probability = coeff * power * exp_term
    return round(probability, 3)