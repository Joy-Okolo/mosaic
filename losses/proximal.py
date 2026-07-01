import torch


def proximal_loss(local_params, global_params, mu):
    """
    FedProx proximal term.
    Penalizes local model for drifting too far from global model.

    Args:
        local_params:  list of local model parameter tensors
        global_params: list of global model parameter tensors
        mu:            proximal strength (from config)
    Returns:
        scalar loss value
    """
    prox = 0.0
    for local_p, global_p in zip(local_params, global_params):
        prox += torch.sum((local_p - global_p.detach()) ** 2)
    return (mu / 2.0) * prox
