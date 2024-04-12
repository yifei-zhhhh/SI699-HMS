"""Grad cam implementation.
"""

import numpy as np
import cv2
from exp import predict_helper


def compute_grad_cam(grads_val, fmap):
    """Compute the grad cam.

    Args:
        grads_val (numpy.ndarray): The gradients.
        fmap (numpy.ndarray): The feature map.

    Returns:
        tuple: The heatmap and the cam.
    """
    H, W = 128, 256
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    grads = grads_val.reshape([grads_val.shape[0], -1])
    weights = np.mean(grads, axis=1)
    for i, w in enumerate(weights):
        cam += w * fmap[i, :, :]
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    return heatmap, cam


def get_grad_cam(model, cnn_layer, batch, target_class):
    """Grad cam implementation.

    Args:
        model (torch.nn.Module): The model to be used.
        cnn_layer (torch.nn.Module): The CNN layer to be used.
        batch (torch.Tensor): The input batch.
        target_class (int): The target class.

    Returns:
        tuple: The heatmap and the cam.
    """
    model.eval()
    grad_block = []
    fmap_block = []

    def backward_hook(module, grad_in, grad_out):
        grad_block.append(grad_out[0].detach())

    def farward_hook(module, input, output):
        fmap_block.append(output)

    h1 = cnn_layer.features[3].register_forward_hook(farward_hook)
    h2 = cnn_layer.features[3].register_backward_hook(backward_hook)
    logits = predict_helper(model, batch)
    model.zero_grad()
    logits[0, target_class].backward()
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    h1.remove()
    h2.remove()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    return compute_grad_cam(grads_val, fmap)
