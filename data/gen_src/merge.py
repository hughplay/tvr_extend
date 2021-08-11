import argparse
import json
import jsonlines
from pathlib import Path
from multiprocessing import Pool

# from rich.progress import track
from tqdm import tqdm


def rename(old_name):
    if old_name.startswith(('img', 'seg')):
        return old_name
    else:
        head, tail = old_name.split('-')
        new_head = '_'.join(head.split('_')[-2:])
        stage, cam, prefix = tail.split('.')
        if stage == 'initial':
            new_stage = 'init'
        elif stage == 'final':
            new_stage = 'fin'
        else:
            new_stage = stage
        new_cam = cam.split('_')[1][0]
        new_tail = '.'.join([new_stage, new_cam, prefix])
        new_name = '-'.join([new_head, new_tail])
        return new_name


def simplfy(args):
    path, mode = args
    with open(path, 'r') as f:
        info = json.load(f)
    if mode == 'full':
        return info
    elif mode == 'simple':
        simple_info = {
            'idx': info['idx'],
            'states': [
                {
                    'objects': [
                        {
                            'color': o['color'],
                            'material': o['material'],
                            'shape': o['shape'],
                            'size': o['size'],
                            'position': o['position']
                        }
                        for o in s['objects']
                    ],
                    'images': {
                        k: rename(v)
                        for k, v in s['images'].items()
                    },
                }
                for s in [info['states'][0], info['states'][-1]]
            ],
            'transformations': [
                {
                    'obj_idx': t['obj_idx'],
                    'attr': t['attr'],
                    'val': t['target'],
                    'options': t['options'] if 'options' in t \
                        else [t['target']],
                    'type': t['type']
                } if t['attr'] == 'position' else {
                    'obj_idx': t['obj_idx'],
                    'attr': t['attr'],
                    'val': t['target'],
                    'options': t['options'] if 'options' in t \
                        else [t['target']],
                } for t in info['transformations']
            ]
        }
    elif mode == 'test':
        simple_info = {
            'idx': info['idx'],
            'states': [
                {
                    'objects': [
                        {
                            'color': o['color'],
                            'material': o['material'],
                            'shape': o['shape'],
                            'size': o['size'],
                            'position': o['position']
                        }
                        for o in s['objects']
                    ],
                    'images': {
                        k: rename(v)
                        for k, v in s['images'].items()
                    },
                }
                for s in [info['states'][0]]
            ]
        }
    elif mode == 'renderable':
        simple_info = {
            'idx': info['idx'],
            'states': [
                {
                    'objects': [
                        {
                            'color': o['color'],
                            'material': o['material'],
                            'shape': o['shape'],
                            'size': o['size'],
                            'position': o['position'],
                            'rotation': o['rotation'],
                        }
                        for o in s['objects']
                    ],
                    'images': {
                        k: rename(v)
                        for k, v in s['images'].items()
                    } if (i == 0 or i == (len(info['states']) - 1)) else {}
                }
                for i, s in enumerate(info['states'])
            ],
            'transformations': [
                {
                    'obj_idx': t['obj_idx'],
                    'attr': t['attr'],
                    'val': t['target'],
                    'options': t['options'] if 'options' in t \
                        else [t['target']],
                    'type': t['type']
                } if t['attr'] == 'position' else {
                    'obj_idx': t['obj_idx'],
                    'attr': t['attr'],
                    'val': t['target'],
                    'options': t['options'] if 'options' in t \
                        else [t['target']],
                } for t in info['transformations']
            ],
            'lamps': info['lamps'],
            'cameras': info['cameras']
        }
    else:
        raise NotImplementedError
    return simple_info



def main(infodir, output='', mode='simple'):
    infofiles = sorted(Path(infodir).glob('*.json'))[:530000]
    args = [(path, mode) for path in infofiles]

    with Pool() as pool:
        samples = list(
            tqdm(pool.imap(simplfy, args), total=len(args), ncols=80))
    with jsonlines.open(output, 'w') as f:
        f.write_all(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infodir')
    parser.add_argument('output')
    parser.add_argument(
        '--mode', default='simple',
        choices=['simple', 'renderable', 'full', 'test'])
    args = parser.parse_args()

    main(args.infodir, args.output, args.mode)