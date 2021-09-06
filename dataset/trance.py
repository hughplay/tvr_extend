import json
import pickle
import random
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomAffine
from pycocotools.mask import toBbox
from pytorch_lightning import LightningDataModule


from .vectorizer import Vectorizer
from .collate import box_collate


FINAL_VIEWS = ['Camera_Center', 'Camera_Left', 'Camera_Right']


class TRANCE(Dataset):

    def __init__(
            self, data_root='/data/trance', split='train',
            values_json='/data/trance/resource/values.json',
            properties_json='/data/trance/resource/properties.json',
            valid_attrs=['position', 'shape', 'size', 'color', 'material'],
            img_aug=False, move_out_aug=False, default_float_type=np.float32,
            use_box=False, box_type='gt'):

        self.ORIGIN_SIZE = (320, 240)

        self.data_root = Path(data_root).expanduser()
        self.data_file = self.data_root / 'data.h5'
        self.split = split
        self.img_aug = img_aug
        self.move_out_aug = move_out_aug
        self.use_box = use_box
        self.box_type = box_type
        assert self.data_file.is_file(), f'{self.data_file} is not existed.'

        self.vectorizer = Vectorizer(
            values_json=values_json, properties_json=properties_json,
            valid_attrs=valid_attrs, default_float_type=default_float_type)

        self.keys = []

    def prepare_trans(self, trans_info):
        obj_idx, pair_idx, options = self.vectorizer.trans2vec(
            trans_info, random_move_out=self.move_out_aug)
        target = (obj_idx, pair_idx)
        return target, options

    def get_boxes(self, objects, view='Camera_Center'):
        if self.box_type == 'gt':
            boxes = [
                toBbox(obj['mask'][view]) for obj in objects
                if obj['mask'][view]['counts'] != 'PP[2']
            random.shuffle(boxes)
        else:
            raise NotImplementedError

        # if len(boxes) == 0:
        #     np.zeros((0, 4), dtype='float32')
        # prepend a full-image box
        boxes.insert(0, [0, 0, *self.ORIGIN_SIZE])
        
        boxes = np.array(boxes)
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        return boxes.astype('float32')

    def prepare_boxes(self, trans_info, final_view='Camera_Center'):
        if self.use_box:
            initial_boxes = self.get_boxes(
                trans_info['states'][0]['objects'], view='Camera_Center')
            final_boxes = self.get_boxes(
                trans_info['states'][-1]['objects'], view=final_view)
            return initial_boxes, final_boxes
        else:
            return None, None

    def prepare_image_and_boxes(
            self, initial_img_name, final_img_name,
            init_boxes=None, fin_boxes=None):
        initial_state = self.read_image(initial_img_name)
        final_state = self.read_image(final_img_name)
        if init_boxes is not None:
            real_size = initial_state.width, initial_state.height
            init_boxes = self.resize_boxes(
                init_boxes, self.ORIGIN_SIZE, real_size)
            fin_boxes = self.resize_boxes(
                fin_boxes, self.ORIGIN_SIZE, real_size)
        if self.img_aug:
            (initial_state, final_state), (init_boxes, fin_boxes) = \
                self.transform(
                    imgs=(initial_state, final_state),
                    boxes=(init_boxes, fin_boxes))
        initial_state, final_state = self.img2arr(initial_state, final_state)
        return initial_state, final_state, init_boxes, fin_boxes

    def read_image(self, name):
        with h5py.File(self.data_file, 'r') as f:
            img = Image.fromarray(f[self.split]['image'][name][()][:,:,:3])
        return img

    def transform(self, imgs, boxes, translation=0.05):
        translate = list(
            translation * (2 * np.random.random(2) - 1) * np.array(
                [imgs[0].width, imgs[0].height]))
        imgs = [
            TF.affine(img, angle=0, translate=translate, scale=1, shear=0)
            for img in imgs]
        if boxes[0] is not None:
            boxes = [
                s_boxes + np.array(
                    [*translate, *translate]).astype(s_boxes.dtype)
                for s_boxes in boxes]
        return imgs, boxes

    def resize_boxes(self, boxes, origin_size, real_size):
        scale_x = real_size[0] / origin_size[0]
        scale_y = real_size[1] / origin_size[1]
        boxes = boxes * np.array(
            [scale_x, scale_y, scale_x, scale_y]).astype(boxes.dtype)
        return boxes

    def img2arr(self, *imgs):
        return [
            np.array(img)[..., :3].transpose([2, 0, 1]) for img in imgs]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        raise NotImplementedError


class Basic(TRANCE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with h5py.File(self.data_file, 'r') as f:
            self.keys = eval(f[self.split]['basic_keys'][()])

    def extract_info(self, idx):
        with h5py.File(self.data_file, 'r') as f:
            sample_info = json.loads(f[self.split]['data'][self.keys[idx]][()])
        init, fin = sample_info['states'][0], sample_info['states'][-1]

        init_img_name = init['images']['Camera_Center']
        fin_img_name = fin['images']['Camera_Center']
        init_desc = init['objects']
        trans = sample_info['transformations'][0]
        init_boxes, fin_boxes = self.prepare_boxes(
            sample_info, final_view='Camera_Center')

        info = {
            'init_img_name': init_img_name,
            'fin_img_name': fin_img_name,
            'init_desc': init_desc,
            'trans': trans,
            'init_boxes': init_boxes,
            'fin_boxes': fin_boxes,
        }
        return info

    def __getitem__(self, idx):
        info = self.extract_info(idx)

        init_img, fin_img, init_boxes, fin_boxes = \
            self.prepare_image_and_boxes(
                info['init_img_name'], info['fin_img_name'],
                info['init_boxes'], info['fin_boxes'])
        init_mat = self.vectorizer.desc2mat(info['init_desc'])
        target, options = self.prepare_trans(info['trans'])

        sample = {
            'init': init_img,
            'fin':fin_img,
            'init_desc': init_mat,
            'target': target,
            'obj_target_vec': init_mat[target[0]],
            "options": options,
            'init_boxes': init_boxes,
            'fin_boxes': fin_boxes,
            'n_init': init_boxes.shape[0] if self.use_box else 0,
            'n_fin': fin_boxes.shape[0] if self.use_box else 0
        }

        return sample


class Event(TRANCE):
    def __init__(self, *args, max_step=4, order_aug=False, **kwargs):
        super().__init__(*args, **kwargs)
        with h5py.File(self.data_file, 'r') as f:
            self.keys = eval(f[self.split]['keys'][()])
        self.final_views = FINAL_VIEWS
        self.order_aug = order_aug
        self.max_step = max_step

    def extract_info(self, idx, final_view='Camera_Center'):
        with h5py.File(self.data_file, 'r') as f:
            sample_info = json.loads(f[self.split]['data'][self.keys[idx]][()])
        init, fin = sample_info['states'][0], sample_info['states'][-1]
        init_boxes, fin_boxes = self.prepare_boxes(
            sample_info, final_view=final_view)

        init_img_name = init['images']['Camera_Center']
        fin_img_name = fin['images'][final_view]
        view_idx = self.final_views.index(final_view)
        init_desc = init['objects']
        fin_desc = fin['objects']
        trans = sample_info['transformations']
        return {
            'init_img_name': init_img_name,
            'fin_img_name': fin_img_name,
            'init_desc': init_desc,
            'fin_desc': fin_desc,
            'trans': trans,
            'view_idx': view_idx,
            'init_boxes': init_boxes,
            'fin_boxes': fin_boxes
        }

    def info2data(self, info):
        init_img, fin_img, init_boxes, fin_boxes = \
            self.prepare_image_and_boxes(
                info['init_img_name'], info['fin_img_name'],
                info['init_boxes'], info['fin_boxes'])
        init_mat = self.vectorizer.desc2mat(info['init_desc'])
        fin_mat = self.vectorizer.desc2mat(info['fin_desc'])

        obj_target_idx, pair_target_idx, obj_target_vec = \
            self.vectorizer.pack_multitrans(
                info['trans'], init_mat, self.max_step,
                random_move_out=self.move_out_aug, random_order=self.order_aug)

        sample = {
            'view': info['view_idx'],
            'init': init_img,
            'fin':fin_img,
            'init_desc': init_mat,
            'fin_desc': fin_mat,
            'target': (obj_target_idx, pair_target_idx),
            'obj_target_vec': obj_target_vec,
            'init_boxes': init_boxes,
            'fin_boxes': fin_boxes,
            'n_init': init_boxes.shape[0] if self.use_box else 0,
            'n_fin': fin_boxes.shape[0] if self.use_box else 0
        }
        return sample

    def __getitem__(self, idx):
        info = self.extract_info(idx)
        return self.info2data(info)


class View(Event):
    def __len__(self):
        return len(self.keys) * len(self.final_views)

    def __getitem__(self, idx):
        sample_idx = idx // len(self.final_views)
        final_view = self.final_views[idx % len(self.final_views)]
        info = self.extract_info(sample_idx, final_view=final_view)
        return self.info2data(info)


class TRANCEDataModule(LightningDataModule):
    def __init__(
            self, name='basic', batch_size=32, num_workers=6, shuffle=True,
            pin_memory=False, img_aug=True, move_out_aug=True,
            use_box=False, box_type='gt', **kwargs):

        super().__init__()

        self.kwargs = kwargs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory

        self.img_aug = img_aug
        self.move_out_aug = move_out_aug

        self.use_box = use_box
        self.box_type = box_type

        self.collate_fn = box_collate if self.use_box else None

        self.final_views = FINAL_VIEWS

        if 'basic' in name:
            self.data_cls = Basic
        elif 'event' in name:
            self.data_cls = Event
        elif 'view' in name:
            self.data_cls = View
        else:
            raise NotImplementedError
    
    def train_dataloader(self):
        train_dataset = self.data_cls(
            split='train', img_aug=self.img_aug,
            move_out_aug=self.move_out_aug, use_box=self.use_box,
            box_type=self.box_type, **self.kwargs)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = self.data_cls(
            split='val', img_aug=False, move_out_aug=False,
            use_box=self.use_box, box_type=self.box_type, **self.kwargs)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
        return val_loader

    def test_dataloader(self):
        test_dataset = self.data_cls(
            split='test', img_aug=False, move_out_aug=False,
            use_box=self.use_box, box_type=self.box_type, **self.kwargs)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )
        return test_loader