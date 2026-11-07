"""Region geometry and complementary prediction masks on the latent grid."""
import math
import torch
import torch.nn.functional as F

from .prompting import DEFAULT_BOXES, parse_boxes


def input_mask(value, height, width):
    mask = value.detach().to(device='cpu', dtype=torch.float32)
    if mask.ndim == 2:
        mask = mask[None]
    if mask.ndim != 3 or not mask.shape[0] or not mask.shape[-1] or not mask.shape[-2]:
        raise ValueError('Connect a MASK with shape H×W or B×H×W.')
    if not bool(torch.isfinite(mask).all()):
        raise ValueError('Masks must contain finite values.')
    return F.interpolate(mask[:1, None], size=(height, width), mode='area')[0, 0].clamp(0, 1)


def rectangle(rect, height, width):
    # Snap to Z-Image's 2×2 latent patch grid, 16 image pixels.
    x, y, w, h = (rect[k] for k in ('x', 'y', 'w', 'h'))
    cols, rows = width // 2, height // 2
    x0, y0 = min(cols - 1, round(x * cols)), min(rows - 1, round(y * rows))
    x1, y1 = max(x0 + 1, round((x + w) * cols)), max(y0 + 1, round((y + h) * rows))
    mask = torch.zeros(height, width)
    mask[y0 * 2:min(height, y1 * 2), x0 * 2:min(width, x1 * 2)] = 1
    return mask


def feather(mask, pixels):
    sigma = pixels / 8 / 2
    if sigma <= 0:
        return mask
    radius = max(1, math.ceil(sigma * 3))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = (-x.square() / (2 * sigma * sigma)).exp()
    kernel /= kernel.sum()
    m = F.pad(mask[None, None], (radius, radius, 0, 0), mode='replicate')
    m = F.conv2d(m, kernel[None, None, None, :])
    m = F.pad(m, (0, 0, radius, radius), mode='replicate')
    return F.conv2d(m, kernel[None, None, :, None])[0, 0]


def layout_masks(height, width, direction, percent, boxes, first_mask, second_mask,
                 edge_softness, separation):
    if direction == 'Free boxes':
        masks = [rectangle(r, height, width) for r in parse_boxes(boxes or DEFAULT_BOXES)]
        layout = 'Free boxes: ' + (boxes or DEFAULT_BOXES)
    else:
        from .composer import region_masks
        masks = list(region_masks(height, width, direction, percent, 'Strong'))
        layout = f'{direction}, {percent:g}/{100-percent:g}'
    for index, value in enumerate((first_mask, second_mask)):
        if value is not None:
            masks[index] = input_mask(value, height, width)
            layout += f'; Character {index+1} mask input'
    # Character 1 has priority in overlapping areas. Keep background unassigned.
    first, second = masks
    if separation == 'Strong':
        first, second = ((m > .5).float() for m in (first, second))
    second = second * (1 - first)
    background = (1 - first - second).clamp(0, 1)
    for number, mask in enumerate((first, second), 1):
        patches = F.avg_pool2d(mask[None, None], 2)
        if not bool((patches > .5).any()):
            raise ValueError(f'Character {number} needs an area of its own on the 16px grid. Enlarge/brighten the mask or reduce overlap.')
    # These weights blend whole regional predictions, including prompts and LoRAs.
    softness = edge_softness if separation == 'Balanced' else 0
    weights = torch.stack([feather(m, softness) for m in (first, second, background)])
    weights /= weights.sum(0).clamp_min(1e-12)
    return weights, layout


def centroid(mask):
    total = mask.sum()
    h, w = mask.shape
    return (float((mask.sum(0) * (torch.arange(w) + .5)).sum() / total / w),
            float((mask.sum(1) * (torch.arange(h) + .5)).sum() / total / h))
