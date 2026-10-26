import torch

def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    # Keep words separately: distinct initial entries may become equal after merges.
    words = [(sequence.split(), freq) for sequence, freq in corpus.items()]
    merges = []

    for _ in range(num_merges):
        pair_to_id = {}
        pair_ids = []
        weights = []

        # Pair IDs are assigned in scan order, so argmax resolves ties correctly.
        for tokens, freq in words:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])

                if pair not in pair_to_id:
                    pair_to_id[pair] = len(pair_to_id)

                pair_ids.append(pair_to_id[pair])
                weights.append(freq)

        if not pair_ids:
            break

        counts = torch.bincount(
            torch.tensor(pair_ids, dtype=torch.long),
            weights=torch.tensor(weights, dtype=torch.float),
            minlength=len(pair_to_id),
        )

        best_id = torch.argmax(counts).item()
        best_pair = next(pair for pair, idx in pair_to_id.items() if idx == best_id)
        merges.append(best_pair)

        left, right = best_pair
        merged_token = left + right

        # Standard BPE: merge matching pairs left-to-right, without overlap.
        updated_words = []
        for tokens, freq in words:
            new_tokens = []
            i = 0

            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            updated_words.append((new_tokens, freq))

        words = updated_words

    return merges