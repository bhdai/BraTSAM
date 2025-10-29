import torch
from tqdm import tqdm


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
