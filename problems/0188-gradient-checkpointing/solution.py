
def checkpoint_forward(funcs, input_arr):
    """
    Applies a list of functions in sequence to the input array, simulating
    gradient checkpointing by not storing intermediates.

    Args:
        funcs (list of callables): List of functions to apply in sequence.
        input_arr (np.ndarray): Input numpy array.

    Returns:
        np.ndarray: The output after applying all functions, cast to float.
    """
    # Ensure numpy array and float dtype
    out = np.asarray(input_arr, dtype=float)

    # Apply each function sequentially without storing intermediate activations
    for f in funcs or []:
        out = f(out)

    # Ensure the returned array is float dtype
    return np.asarray(out, dtype=float)
