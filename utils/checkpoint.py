import torch, os

def save_checkpoint(path, model, optimizer, iteration, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "extra": extra or {},
    }
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None, map_location="cuda"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state["iteration"], state.get("extra", {})