from bisect import bisect_right

def progressive_batch_size(
    queries: list[int],
    milestones: list[int],
    batch_sizes: list[int]
) -> list[int]:
    
    return [
        batch_sizes[bisect_right(milestones, tokens)]
        for tokens in queries
    ]