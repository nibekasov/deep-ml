def partition_parameters(params):
    """
    Partition parameter names into Muon and AdamW groups.
    """
    result = {
        "muon": [],
        "adamw": []
    }

    for name, shape in params:
        if len(shape) == 2 and "embed" not in name:
            result["muon"].append(name)
        else:
            result["adamw"].append(name)

    return result