import torch


def save_model_checkpoint(path, epoch, model, ema_model, optimizer, losses):
    model_checkpoint = {
        "current_epoch": epoch,
        "model_state": model.state_dict(),
        "ema_model_state": ema_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss_history": losses,
    }
    torch.save(model_checkpoint, path)


def load_checkpoint(checkpoint_path):
    model_checkpoint = torch.load(checkpoint_path)
    epoch = model_checkpoint["current_epoch"]
    model_weights = model_checkpoint["model_state"]
    ema_model_weights = model_checkpoint["ema_model_state"]
    optimizer_state = model_checkpoint["optimizer_state"]
    loss_hist = model_checkpoint["loss_history"]
    return epoch, model_weights, ema_model_weights, optimizer_state, loss_hist
