import json
import os
from pathlib import Path

import torch


class ActivityNet(torch.utils.data.Dataset):

    def __init__(self, root, split):
        super(ActivityNet, self).__init__()
        self.split = split
        self.root = root

        with open(os.path.join(self.root, 'anet.json')) as f:
            self.ann = json.load(f)

        self.feat_path = os.path.join(self.root, 'merge')
        self.video_idx = self.build_video_idx()
        self.ann = self.revise_label_and_duration()

        self._get_item = getattr(self, f'_get_item_{self.split}')

    def build_video_idx(self):
        """If feat is shorter than `length`, pad it with a learnable vector to
        `length`. If feat is longer than `length`, remove it for training and 
        validation; for testing, divide it into multiple chuncks with `length`.
        """
        video_idx = [k.split('.')[0] for k in os.listdir(self.feat_path)]
        video_split = [k for k in video_idx
            if self.split in self.ann['database'][k]['subset']]
        return video_split

    def revise_label_and_duration(self):
        self.name2label = dict()
        self.label2name = dict()
        count = 0
        for k, v in self.ann['database'].items():
            for item in v['annotations']:
                label = item['label']
                if label not in self.name2label.keys():
                    self.name2label[label] = count
                    count += 1
        
        for k, v in self.name2label.items():
            self.label2name[v] = k

        with open(os.path.join(self.root, 'duration.json')) as f:
            duration = json.load(f)

        new_ann = dict()
        for k, v in self.ann['database'].items():
            new_ann[k] = {'annotations': []}
            for item in v['annotations']:
                new_ann[k]['annotations'].append({
                    'label': self.name2label[item['label']],
                    'segment': [min(item['segment'][0], duration[k]),
                                min(item['segment'][1], duration[k])], 
                })

        return new_ann

    def _get_item_train(self, index):
        """
        Return:
            feat: torch.FloatTensor (?, 2304)
            targets: {segment: tuple, label: int, }
        """
        video_id = self.video_idx[index]
        feat_path = os.path.join(self.feat_path, f'{video_id}.pth')
        feat = torch.load(feat_path)
        targets = self.ann[video_id]['annotations']

        labels = [item['label'] for item in targets]
        labels = torch.as_tensor(labels, dtype=torch.long)

        segments = list()
        for k in targets:
            center = (k['segment'][0] + k['segment'][1]) / 2.0
            length = k['segment'][1] - k['segment'][0]
            segments.append([center, length])

        segments = torch.as_tensor(segments, dtype=torch.float32)
        segments = segments / feat.size(0)

        targets = {'boxes': segments, 'labels': labels}
        return feat, targets

    def _get_item_val(self, index):
        video_id = self.video_idx[index]
        feat_path = os.path.join(self.feat_path, f'{video_id}.pth')
        feat = torch.load(feat_path)
        return feat, index

    def __getitem__(self, index):
        return self._get_item(index)

    def __len__(self):
        return len(self.video_idx)


def build(video_set, args):
    root = Path(args.anet_path)
    assert root.exists(), f'provided anet path {root} does not exist'

    dataset = ActivityNet(root, video_set)
    return dataset
