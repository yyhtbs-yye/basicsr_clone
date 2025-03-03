import cv2
import random
import torch
import numpy as np

def mod_crop(img, scale):
    """Mod crop images, used during testing.

    Args:
        img (ndarray): Input image.
        scale (int): Scale factor.

    Returns:
        ndarray: Result image.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[0], img.shape[1]
        h_remainder, w_remainder = h % scale, w % scale
        img = img[:h - h_remainder, :w - w_remainder, ...]
    else:
        raise ValueError(f'Wrong img ndim: {img.ndim}.')
    return img


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None, padding_mode='constant'):
    """Paired random crop with padding. Support Numpy array and Tensor inputs.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images.
        img_lqs (list[ndarray] | ndarray): LQ images.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.
        padding_mode (str): Type of padding (e.g., 'constant', 'reflect'). Default: 'reflect'.

    Returns:
        tuple: GT images and LQ images.
    """

    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[:2]
        h_gt, w_gt = img_gts[0].shape[:2]

    lq_patch_size = gt_patch_size // scale

    # Calculate required padding for LQ and GT
    pad_lq_h = max(0, lq_patch_size - h_lq)
    pad_lq_w = max(0, lq_patch_size - w_lq)
    pad_gt_h = max(0, gt_patch_size - h_gt)
    pad_gt_w = max(0, gt_patch_size - w_gt)

    # Pad LQ and GT images if necessary
    if pad_lq_h > 0 or pad_lq_w > 0:
        if input_type == 'Tensor':
            pad = (pad_lq_w // 2, pad_lq_w - pad_lq_w // 2, pad_lq_h // 2, pad_lq_h - pad_lq_h // 2)
            img_lqs = [torch.nn.functional.pad(v, pad, mode=padding_mode) for v in img_lqs]
        else:
            img_lqs = [np.pad(v, ((pad_lq_h // 2, pad_lq_h - pad_lq_h // 2), (pad_lq_w // 2, pad_lq_w - pad_lq_w // 2), (0, 0)), mode=padding_mode) for v in img_lqs]

    if pad_gt_h > 0 or pad_gt_w > 0:
        if input_type == 'Tensor':
            pad = (pad_gt_w // 2, pad_gt_w - pad_gt_w // 2, pad_gt_h // 2, pad_gt_h - pad_gt_h // 2)
            img_gts = [torch.nn.functional.pad(v, pad, mode=padding_mode) for v in img_gts]
        else:
            img_gts = [np.pad(v, ((pad_gt_h // 2, pad_gt_h - pad_gt_h // 2), (pad_gt_w // 2, pad_gt_w - pad_gt_w // 2), (0, 0)), mode=padding_mode) for v in img_gts]

    # Update dimensions after padding
    h_lq, w_lq = (h_lq + pad_lq_h, w_lq + pad_lq_w) if input_type == 'Numpy' else (img_lqs[0].size(-2), img_lqs[0].size(-1))
    h_gt, w_gt = (h_gt + pad_gt_h, w_gt + pad_gt_w) if input_type == 'Numpy' else (img_gts[0].size(-2), img_gts[0].size(-1))

    # Crop images
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    if input_type == 'Tensor':
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
        img_gts = [v[:, :, top * scale:top * scale + gt_patch_size, left * scale:left * scale + gt_patch_size] for v in img_gts]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]
        img_gts = [v[top * scale:top * scale + gt_patch_size, left * scale:left * scale + gt_patch_size, ...] for v in img_gts]

    # Handle single-image input
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]

    return img_gts, img_lqs

def paired_center_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired center crop. Supports Numpy array and Tensor inputs.

    It crops lists of LQ and GT images from the center, ensuring corresponding locations.

    Args:
        img_gts (list[ndarray] | ndarray | list[Tensor] | Tensor): GT images.
            All images should have the same shape. If the input is an ndarray, it will
            be transformed into a list containing itself.
        img_lqs (list[ndarray] | ndarray): LQ images.
            All images should have the same shape. If the input is an ndarray, it will
            be transformed into a list containing itself.
        gt_patch_size (int): GT patch size.
        scale (int): Scale factor.
        gt_path (str): Path to ground-truth. Default: None.

    Returns:
        list[ndarray] | ndarray: Cropped GT images and LQ images. If the result
            contains only one element, an ndarray is returned directly.
    """
    # Convert single image inputs to lists
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    # Determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(img_gts[0]) else 'Numpy'

    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[:2]
        h_gt, w_gt = img_gts[0].shape[:2]

    # Compute $lq_patch_size = gt_patch_size / scale$
    lq_patch_size = gt_patch_size // scale

    # Check if padding is needed for LQ and GT images
    pad_height_lq = max(0, lq_patch_size - h_lq)
    pad_width_lq = max(0, lq_patch_size - w_lq)
    pad_height_gt = max(0, gt_patch_size - h_gt)
    pad_width_gt = max(0, gt_patch_size - w_gt)

    # Pad LQ images if needed
    if pad_height_lq > 0 or pad_width_lq > 0:
        if input_type == 'Tensor':
            padding_lq = (pad_width_lq // 2, pad_width_lq - pad_width_lq // 2, pad_height_lq // 2, pad_height_lq - pad_height_lq // 2)
            img_lqs = [torch.nn.functional.pad(v, padding_lq, "constant", 0) for v in img_lqs]
        else:
            img_lqs = [np.pad(v, ((pad_height_lq // 2, pad_height_lq - pad_height_lq // 2),
                                  (pad_width_lq // 2, pad_width_lq - pad_width_lq // 2),
                                  (0, 0)), 'constant') for v in img_lqs]

    # Pad GT images if needed
    if pad_height_gt > 0 or pad_width_gt > 0:
        if input_type == 'Tensor':
            padding_gt = (pad_width_gt // 2, pad_width_gt - pad_width_gt // 2, pad_height_gt // 2, pad_height_gt - pad_height_gt // 2)
            img_gts = [torch.nn.functional.pad(v, padding_gt, "constant", 0) for v in img_gts]
        else:
            img_gts = [np.pad(v, ((pad_height_gt // 2, pad_height_gt - pad_height_gt // 2),
                                  (pad_width_gt // 2, pad_width_gt - pad_width_gt // 2),
                                  (0, 0)), 'constant') for v in img_gts]


    if input_type == 'Tensor':
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]


    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x '
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # Compute center crop coordinates
    top_lq = (h_lq - lq_patch_size) // 2
    left_lq = (w_lq - lq_patch_size) // 2
    top_gt = (h_gt - gt_patch_size) // 2
    left_gt = (w_gt - gt_patch_size) // 2

    # Crop the images
    if input_type == 'Tensor':
        img_lqs = [v[:, :, top_lq:top_lq + lq_patch_size, left_lq:left_lq + lq_patch_size] for v in img_lqs]
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_lqs = [v[top_lq:top_lq + lq_patch_size, left_lq:left_lq + lq_patch_size, ...] for v in img_lqs]
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]

    # Return single image if only one element is present
    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]

    return img_gts, img_lqs


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs


def img_rotate(img, angle, center=None, scale=1.0):
    """Rotate image.

    Args:
        img (ndarray): Image to be rotated.
        angle (float): Rotation angle in degrees. Positive values mean
            counter-clockwise rotation.
        center (tuple[int]): Rotation center. If the center is None,
            initialize it as the center of the image. Default: None.
        scale (float): Isotropic scale factor. Default: 1.0.
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(img, matrix, (w, h))
    return rotated_img
