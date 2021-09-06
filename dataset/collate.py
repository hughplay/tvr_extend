import torch
from torch.utils.data.dataloader import default_collate
from torchvision.ops._utils import convert_boxes_to_roi_format


BOX_KEYWORD = 'boxes'


def box_collate(batch):
    elem = batch[0]
    collate_batch = {
        key: default_collate([d[key] for d in batch])
        for key in elem if BOX_KEYWORD not in key
    }
    collate_batch_boxes = {
        key: convert_boxes_to_roi_format(
            [torch.as_tensor(d[key]) for d in batch])
        for key in elem if BOX_KEYWORD in key
    }
    collate_batch.update(collate_batch_boxes)
    return collate_batch