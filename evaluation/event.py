import json
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F

from dataset.vectorizer import Vectorizer


class EventEvaluator:

    def __init__(
            self, values_json='/data/trance/resource/values.json',
            properties_json='/data/trance/resource/properties.json',
            valid_attrs=['position', 'shape', 'size', 'color', 'material']):

        self.t = Vectorizer(
            values_json=values_json, properties_json=properties_json,
            valid_attrs=valid_attrs)
        self.tensor_size = torch.tensor([self.t.coord[s] for s in self.t.size])

        self.EPSILON = 1e-4

    def compute_loss(self, obj_choice, pair_choice, obj_target, pair_target):
        B, L, _ = obj_choice.shape
        loss_obj = F.nll_loss(
            obj_choice.view(B * L, -1), obj_target.view(B * L),
            ignore_index=self.t.OBJ_PAD)
        loss_pair = F.nll_loss(
            pair_choice.view(B * L, -1), pair_target.view(B * L),
            ignore_index=self.t.PAIR_PAD)
        loss = loss_obj + loss_pair
        return loss, loss_obj, loss_pair

    def evaluate(
            self, obj_choice, pair_choice, init_desc, fin_desc, 
            obj_target, pair_target, detail=False):

        loss, loss_obj, loss_pair = self.compute_loss(
            obj_choice, pair_choice, obj_target, pair_target)

        preds = (obj_choice.argmax(dim=2), pair_choice.argmax(dim=2))
        targets = (obj_target, pair_target)

        res = self.eval_tensor_results(
            preds, init_desc, fin_desc, targets, keep_tensor=True)

        n_sample = obj_choice.shape[0]
        info ={
            'n_sample': n_sample,
            'loss': loss.item() * n_sample,
            'loss_obj': loss_obj.item() * n_sample,
            'loss_pair': loss_pair.item() * n_sample,
            'acc': res['correct'].sum().item(),
            'loose_acc': res['loose_correct'].sum().item(),
            'avg_dist': res['dist'].sum().item(),
            'avg_norm_dist': res['norm_dist'].sum().item(),
            'avg_step_diff': (
                res['pred_step'] - res['target_step']).sum().item()
        }

        if detail:
            res = {
                k : v.squeeze().tolist() if type(v) is torch.Tensor else v
                for k, v in res.items()}
            info['detail'] = res

        return loss, info

    def eval_text_results(self, predictions, samples, keep_tensor=False):
        init_mat_batch, fin_mat_batch = [], []
        targets, preds = [], []
        for sample, prediction in zip(samples, predictions):
            init_mat_batch.append(
                self.t.desc2mat(sample['states'][0]['objects']))
            fin_mat_batch.append(
                self.t.desc2mat(sample['states'][-1]['objects']))

            targets.append(self.t.multitrans2vec(sample['transformations']))
            preds.append(self.t.multitrans2vec(prediction))
        init_mat_batch = torch.from_numpy(np.stack(init_mat_batch))
        fin_mat_batch = torch.from_numpy(np.stack(fin_mat_batch))
        return self.eval_tensor_results(
            preds, init_mat_batch, fin_mat_batch, targets, keep_tensor)

    def eval_text_result(self, sample, prediction):
        res = self.eval_text_results([prediction], [sample])
        return {k: v[0] for k, v in res.items()}

    def eval_tensor_results(
            self, preds, init_mat_batch, fin_mat_batch, targets,
            keep_tensor=False):
        B = init_mat_batch.shape[0]
        device = init_mat_batch.device

        if type(preds[0]) is torch.Tensor:
            preds = self.unpack_batch(*preds)
        if type(targets[0]) is torch.Tensor:
            targets = self.unpack_batch(*targets)

        error_overlap = torch.full((B,), False, dtype=torch.bool).to(device)
        error_out = torch.full((B,), False, dtype=torch.bool).to(device)

        for i, (init_mat, pred) in enumerate(zip(init_mat_batch, preds)):
            error_overlap[i], error_out[i] = self.transform(init_mat, pred)

        # init_mat_batch has been transformed
        dist = self.compute_diff(init_mat_batch, fin_mat_batch)
        loose_correct = (dist == 0)
        pred_step = torch.tensor([len(p) for p in preds]).to(device)
        target_step = torch.tensor([len(t) for t in targets]).to(device)

        res = {
            'dist': dist,
            'norm_dist': 1. * dist / target_step,
            'loose_correct': loose_correct,
            'err_overlap': error_overlap,
            'err_invalid_position': error_out,
            'correct': loose_correct & ~error_overlap & ~error_out,
            'pred': preds,
            'target': targets,
            'pred_step': pred_step,
            'target_step': target_step
        }

        if not keep_tensor:
            res = {
                k : v.tolist() if type(v) is torch.Tensor else v
                for k, v in res.items()}

        return res

    def unpack_batch(self, obj_batch, pair_batch):
        res = []
        for objs, pairs in zip(obj_batch, pair_batch):
            sample = []
            for obj, pair in zip(objs, pairs):
                if obj >= self.t.OBJ_EOS or pair >= self.t.PAIR_EOS:
                    break
                sample.append((obj.item(), pair.item()))
            res.append(sample)
        return res

    def compute_diff(self, pred_fin_mat, target_fin_mat):
        diff = torch.abs(pred_fin_mat - target_fin_mat) > self.EPSILON

        pos_pred = self.t.restore_position(pred_fin_mat[
            ..., self.t.feat_start['position']:self.t.feat_end['position']])
        pos_target = self.t.restore_position(target_fin_mat[
            ..., self.t.feat_start['position']:self.t.feat_end['position']])
        pos_equal = torch.sum(
            torch.abs(pos_pred - pos_target) < self.EPSILON, dim=2) \
                == pos_pred.shape[2]

        vis_pred = self.b_is_visible(pos_pred)
        vis_target = self.b_is_visible(pos_target)

        pos_equal = ((vis_target == Visibility.visible) & pos_equal) | (
            (vis_target == Visibility.invisible)
            & (vis_pred == Visibility.invisible))
        diff[..., self.t.feat_start['position']:self.t.feat_end['position']] \
            = ~pos_equal[..., None]

        n_diff = torch.sum(diff, dim=(1, 2)) / 2

        return n_diff

    def transform(self, init_mat, pred):
        error_overlap = error_out = False
        for obj_idx, pair_idx in pred:
            e_overlap, e_out = \
                self.transform_step(init_mat, obj_idx, pair_idx)
            error_overlap |= e_overlap
            error_out |= e_out
        return error_overlap, error_out

    def transform_step(self, objs, obj_idx, pair_idx):
        attr, value = self.t.sep_pair(pair_idx)
        error_overlap = error_out = False
        obj = objs[obj_idx]

        func = getattr(self, 't_{}'.format(attr))
        func(obj, value)

        if attr == 'position':
            after = self.is_visible(
                obj[self.t.feat_start['position']:self.t.feat_end['position']])
            if after is Visibility.invalid:
                error_out = True

        if attr in ['position', 'size']:
            error_overlap = self.is_overlap(objs, obj_idx)

        return error_overlap, error_out

    def t_position(self, obj, target):
        direction, step = target
        v_direction = self.t.properties['position']['direction'][direction]

        obj[self.t.feat_start['position']:self.t.feat_end['position']] += \
            (torch.tensor(v_direction).to(obj) * step * self.t.coord['step'] \
                    / (self.t.pos_max - self.t.pos_min))

    def t_material(self, obj, material):
        obj[self.t.feat_start['material']:self.t.feat_end['material']] = 0
        obj[self.t.feat_start['material'] + self.t.material.index(
            material)] = 1

    def t_color(self, obj, color):
        obj[self.t.feat_start['color']:self.t.feat_end['color']] = 0
        obj[self.t.feat_start['color'] + self.t.color.index(color)] = 1

    def t_shape(self, obj, shape):
        obj[self.t.feat_start['shape']:self.t.feat_end['shape']] = 0
        obj[self.t.feat_start['shape'] + self.t.shape.index(shape)] = 1

    def t_size(self, obj, size):
        obj[self.t.feat_start['size']:self.t.feat_end['size']] = 0
        obj[self.t.feat_start['size'] + self.t.size.index(size)] = 1

    def is_overlap(self, state_mat, obj_idx):
        pos = state_mat[
            :, self.t.feat_start['position']:self.t.feat_end['position']]
        size_idx = torch.argmax(
            state_mat[:, self.t.feat_start['size']:self.t.feat_end['size']],
            dim=1)
        size = self.tensor_size.to(size_idx)[size_idx]

        obj_size = size[obj_idx]
        obj_pos = pos[obj_idx]

        dist = torch.norm(pos - obj_pos, dim=1) * (
            self.t.pos_max - self.t.pos_min)
        min_dist = size + obj_size + self.t.coord['min_gap']

        n_violate = torch.sum(dist + self.EPSILON < min_dist.to(dist))
        res = (n_violate > 1).item()

        return res

    def is_visible(self, pos):
        pos = self.t.restore_position(pos)
        x = pos[..., 0]
        y = pos[..., 1]
        if (self.t.coord['x_min'] <= x <= self.t.coord['x_max'] \
                and self.t.coord['y_min'] <= y <= self.t.coord['y_max']):
            if (
                    self.t.coord['vis_x_min'] <= x
                    <= self.t.coord['vis_x_max']
                    and self.t.coord['vis_y_min'] <= y
                    <= self.t.coord['vis_y_max']):
                return Visibility.visible
            else:
                return Visibility.invisible
        else:
            return Visibility.invalid

    def b_is_visible(self, pos):
        x = pos[..., 0]
        y = pos[..., 1]

        state = torch.full(
            pos.shape[:-1], Visibility.invisible,
            dtype=torch.uint8).to(pos.device)

        invalid_x = (x > self.t.coord['x_max']) | (x < self.t.coord['x_min'])
        invalid_y = (y > self.t.coord['y_max']) | (y < self.t.coord['y_min'])

        vis_x = (x >= self.t.coord['vis_x_min']) & (
            x <= self.t.coord['vis_x_max'])
        vis_y = (y >= self.t.coord['vis_y_min']) & (
            y <= self.t.coord['vis_y_max'])

        state[invalid_x | invalid_y] = Visibility.invalid
        state[vis_x & vis_y] = Visibility.visible

        return state


class ReinforceEventEvaluator(EventEvaluator):

    def __init__(
            self, reward_type='acc_dist',
            values_json='/data/trance/resource/values.json',
            properties_json='/data/trance/resource/properties.json',
            valid_attrs=['position', 'shape', 'size', 'color', 'material']):

        self.reward_type = reward_type
        super().__init__(values_json, properties_json, valid_attrs)

    def compute_loss(
            self, obj_choice, pair_choice, obj_target, pair_target, reward):
        B, L, _ = obj_choice.shape
        reward = reward[..., None].expand(-1, L).reshape(-1)
        loss_obj = torch.mean(F.nll_loss(
            obj_choice.view(B * L, -1), obj_target.view(B * L),
            ignore_index=self.t.OBJ_PAD, reduction='none') * reward)
        loss_pair = torch.mean(F.nll_loss(
            pair_choice.view(B * L, -1), pair_target.view(B * L),
            ignore_index=self.t.PAIR_PAD, reduction='none') * reward)
        loss = loss_obj + loss_pair
        return loss, loss_obj, loss_pair

    def compute_reward(self, res):
        if self.reward_type == 'acc':
            reward = res['correct'] + 1.
        elif self.reward_type == 'dist':
            reward = 2. - 1. * res['norm_dist']
        elif self.reward_type == 'acc_dist':
            reward = 2. + res['correct'] - 1. * res['norm_dist']
        else:
            raise NotImplementedError

        return reward.detach()

    def evaluate(
            self, obj_choice, pair_choice, init_desc, fin_desc, 
            obj_target, pair_target, detail=False):

        preds = (obj_choice.argmax(dim=2), pair_choice.argmax(dim=2))
        targets = (obj_target, pair_target)
        res = self.eval_tensor_results(
            preds, init_desc, fin_desc, targets, keep_tensor=True)

        reward = self.compute_reward(res)
        loss, loss_obj, loss_pair = self.compute_loss(
            obj_choice, pair_choice, obj_target, pair_target, reward)

        n_sample = obj_choice.shape[0]
        info ={
            'n_sample': n_sample,
            'loss': loss.item() * n_sample,
            'loss_obj': loss_obj.item() * n_sample,
            'loss_pair': loss_pair.item() * n_sample,
            'reward': reward.sum().item(),
            'acc': res['correct'].sum().item(),
            'loose_acc': res['loose_correct'].sum().item(),
            'avg_dist': res['dist'].sum().item(),
            'avg_norm_dist': res['norm_dist'].sum().item(),
            'avg_step_diff': (
                res['pred_step'] - res['target_step']).sum().item()
        }

        if detail:
            res['reward'] = reward
            res = {
                k : v.squeeze().tolist() if type(v) is torch.Tensor else v
                for k, v in res.items()}
            info['detail'] = res

        return loss, info


class Visibility:
    invisible = 0
    visible = 1
    invalid = 2