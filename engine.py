import torch
from tqdm import tqdm


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience"""

    def __init__(
        self,
        patience=20,
        verbose=False,
        delta=0,
        path="best_model.pth",
        latest_path="latest_model.pth",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved
            verbose (bool): if true, prints the message for each validation loss improvement
            delta (float): minimum change int he monitored quanlity to quantify as an improvment
            path (str): path to the best checkpoint to be saved to
            latest_path (str): path to the latest checkpoint to be saved to
            trace_func (function): trace print function
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta
        self.path = path
        self.latest_path = latest_path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        # Always save the latest model
        self.save_latest(model)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """saves model when validation loss decrease"""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def save_latest(self, model):
        """saves the latest model checkpoint"""
        torch.save(model.state_dict(), self.latest_path)


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, scaler=None):
    """
    perform one full training epoch.
    """
    model.train()
    total_loss = 0.0
    use_amp = scaler is not None

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        pixel_values = batch["pixel_values"].to(device)
        input_boxes = batch["input_boxes"].to(device)
        ground_truth_masks = batch["labels"].to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type=device.type):
                predicted_masks = model(
                    pixel_values=pixel_values, input_boxes=input_boxes
                )
                loss = loss_fn(
                    predicted_masks.unsqueeze(1),
                    ground_truth_masks.unsqueeze(1).float(),
                )
            scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            predicted_masks = model(pixel_values=pixel_values, input_boxes=input_boxes)
            loss = loss_fn(
                predicted_masks.unsqueeze(1),
                ground_truth_masks.unsqueeze(1).float(),
            )
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_epoch_loss = total_loss / len(dataloader)
    return avg_epoch_loss


def evaluate(model, dataloader, loss_fn, device, use_amp=False):
    """
    performs validation on the model.
    """
    model.eval()
    total_loss = 0.0

    # disable gradient calculation
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating", leave=False)
        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            input_boxes = batch["input_boxes"].to(device)
            ground_truth_masks = batch["labels"].to(device)

            if use_amp:
                with torch.amp.autocast(device_type=device.type):
                    predicted_masks = model(
                        pixel_values=pixel_values, input_boxes=input_boxes
                    )
                    loss = loss_fn(
                        predicted_masks.unsqueeze(1),
                        ground_truth_masks.unsqueeze(1).float(),
                    )
            else:
                predicted_masks = model(
                    pixel_values=pixel_values, input_boxes=input_boxes
                )
                loss = loss_fn(
                    predicted_masks.unsqueeze(1),
                    ground_truth_masks.unsqueeze(1).float(),
                )

            # update running loss
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    avg_epoch_loss = total_loss / len(dataloader)
    return avg_epoch_loss
