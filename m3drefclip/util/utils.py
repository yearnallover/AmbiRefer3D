import torch


def get_batch_aabb_pair_ious(batch_boxes_1_bound, batch_boxes_2_bound):
    boxes_1 = torch.nan_to_num(batch_boxes_1_bound.float())
    boxes_2 = torch.nan_to_num(batch_boxes_2_bound.float())
    boxes_1 = torch.stack((boxes_1.amin(dim=1), boxes_1.amax(dim=1)), dim=1)
    boxes_2 = torch.stack((boxes_2.amin(dim=1), boxes_2.amax(dim=1)), dim=1)

    box_1_x_min, box_1_y_min, box_1_z_min = torch.tensor_split(boxes_1[:, 0], 3, dim=1)
    box_1_x_max, box_1_y_max, box_1_z_max = torch.tensor_split(boxes_1[:, 1], 3, dim=1)

    box_2_x_min, box_2_y_min, box_2_z_min = torch.tensor_split(boxes_2[:, 0], 3, dim=1)
    box_2_x_max, box_2_y_max, box_2_z_max = torch.tensor_split(boxes_2[:, 1], 3, dim=1)

    x_a = torch.maximum(box_1_x_min, box_2_x_min)
    y_a = torch.maximum(box_1_y_min, box_2_y_min)
    z_a = torch.maximum(box_1_z_min, box_2_z_min)
    x_b = torch.minimum(box_1_x_max, box_2_x_max)
    y_b = torch.minimum(box_1_y_max, box_2_y_max)
    z_b = torch.minimum(box_1_z_max, box_2_z_max)

    zero_tensor = torch.zeros_like(x_a)
    intersection_volume = torch.maximum((x_b - x_a), zero_tensor) * torch.maximum((y_b - y_a), zero_tensor) * \
                          torch.maximum((z_b - z_a), zero_tensor)
    box_1_volume = (box_1_x_max - box_1_x_min) * (box_1_y_max - box_1_y_min) * (box_1_z_max - box_1_z_min)
    box_2_volume = (box_2_x_max - box_2_x_min) * (box_2_y_max - box_2_y_min) * (box_2_z_max - box_2_z_min)
    union = (box_1_volume + box_2_volume - intersection_volume).clamp_min(
        torch.finfo(torch.float32).eps
    )
    iou = intersection_volume / union
    return torch.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0).flatten()
