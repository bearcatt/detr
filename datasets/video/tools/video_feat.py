"""
Ts video clip -> 2s window, 1s stride -> (T-1)x 64 frames

SlowFast 8x8 R50：
    (T-1)x 64 frames -> slowfast -> (T-1)x feature vector
I3D:
    (T-1)x 64 frames -> I3D -> (T-1)x feature vector

python video_feat.py --cfg SLOWFAST_8x8_R50.yaml \
  MODEL.MODEL_NAME SlowFastFeat \
  DATA.PATH_TO_DATA_DIR . \
  TEST.DATASET ActivityNetRaw \
  TEST.BATCH_SIZE 20 \
  TEST.CHECKPOINT_FILE_PATH SLOWFAST_8x8_R50.pkl \
  NUM_GPUS 4 \
  DATA_LOADER.NUM_WORKERS 12
"""

import json
import os
import random
import math

import cv2
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from tqdm import tqdm

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.misc as misc
import slowfast.utils.multiprocessing as mpu
from slowfast.datasets import DATASET_REGISTRY, loader, utils
from slowfast.models import MODEL_REGISTRY, SlowFast, build_model
from slowfast.utils.parser import load_config, parse_args


@MODEL_REGISTRY.register()
class SlowFastFeat(SlowFast):
    def __init__(self, cfg):
        super(SlowFastFeat, self).__init__(cfg)
        self.avg_pool0 = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avg_pool1 = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
    
    def forward(self, x, bboxes=None):
        """
        Return: N x 2304
        """
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)

        x = torch.cat([self.avg_pool0(x[0]), self.avg_pool1(x[1])], 1)
        return x.squeeze()


@DATASET_REGISTRY.register()
class Activitynetraw(torch.utils.data.Dataset):
    def __init__(self, cfg, split, num_retries=10):
        """
        Args:
            cfg (CfgNode): configs.
            num_retries (int): number of retries.
        """
        self.cfg = cfg
        self._num_retries = num_retries

        self._clip_size = 2.13
        self._target_frames = 64
        self._clip_stride = 1
        self._construct_loader()
        self.current_video = ''

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        # TODO: merge into the annotation file, and modify the action duration.
        with open(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, "duration.json")) as f:
            duration_dict = json.load(f)

        self.raw_video_path = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'raw')
        _video_names = PathManager.ls(self.raw_video_path)
        _video_durations = [duration_dict[k] for k in _video_names]

        self._video_names = list()
        self._clip_idx = list()
        self._video_durations = list()
        for i, duration in enumerate(_video_durations):
            if duration >= self._clip_size:
                _num_clips = int(duration - self._clip_size) + 1
                for j in range(_num_clips):
                    self._video_names.append(_video_names[i])
                    self._clip_idx.append(j)
                    self._video_durations.append(duration)

    def sample_frames(self, index):
        """
        Return:
            frames (tersor): a tensor of temporal sampled video frames, dimension is
                `target_frames` x `height` x `width` x `channel`.
        """
        video_frame_path = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, 'frames', self._video_names[index]
        )
        frame_path_list = os.listdir(video_frame_path)

        video_length = len(frame_path_list)
        video_duration = self._video_durations[index]
        fps = video_length / video_duration

        acquired_length = int(math.ceil(fps * self._clip_size))
        start_idx = int(math.floor(fps * self._clip_idx[index]))
        start_idx = min(start_idx, video_length - acquired_length)

        frames = list()
        for i in range(start_idx, start_idx + acquired_length):
            frame = cv2.imread(
                os.path.join(video_frame_path, "{:010d}.jpg".format(i))
            )
            frames.append(torch.from_numpy(frame))

        frames = torch.stack(frames, dim=0)
        idx = torch.linspace(0, frames.shape[0] - 1, self._target_frames)
        idx = torch.clamp(idx, 0, frames.shape[0] - 1).long()
        frames = torch.index_select(frames, 0, idx)
        return frames


    def __getitem__(self, index):
        # Decode video. Meta info is used to perform selective decoding.
        frames = self.sample_frames(index)

        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        frames = utils.pack_pathway_output(self.cfg, frames)

        return frames, index

    def __len__(self):
        return len(self._video_names)


class Saver(object):
    def __init__(self, root, dataset):
        self.save_dir = os.path.join(root, 'feat')
        self.merge_dir = os.path.join(root, 'merge')
        self.video_names = dataset._video_names
        self.clip_idxes = dataset._clip_idx

        self.max_length = 256
        self.min_length = 2
        self.dim = 2304

    def save(self, feats, index):
        bs = index.numel()
        feats = feats.view(bs, -1).detach().cpu().numpy() # N, C
        index = index.cpu().numpy() # N,

        for feat, idx in zip(feats, index):
            video_name = self.video_names[idx].split('.')[0][2:]

            video_dir = os.path.join(self.save_dir, video_name)
            if not PathManager.exists(video_dir):
                PathManager.mkdirs(video_dir)

            clip_idx = self.clip_idxes[idx]
            torch.save(feat, os.path.join(video_dir, f'{clip_idx}.pth'))

    def merge(self):
        """merge all clip features of a video into one/several 
           fix-size matrix(es)
        """
        if not PathManager.exists(self.merge_dir):
            PathManager.mkdirs(self.merge_dir)

        for video_name in PathManager.ls(self.save_dir):
            video_dir = os.path.join(self.save_dir, video_name)
            num_feats = len(PathManager.ls(video_dir))

            if self.min_length <= num_feats <= self.max_length:
                merged_feat = torch.zeros((num_feats, self.dim), dtype=torch.float32)

                for clip_idx in range(num_feats):
                    feat = torch.load(os.path.join(video_dir, f'{clip_idx}.pth'))
                    merged_feat[clip_idx, :] = torch.from_numpy(feat)

                torch.save(merged_feat, os.path.join(self.merge_dir, f'{video_name}.pth'))
            else:
                # TODO
                print(video_name)


def inference(cfg):
    # # Set up environment.
    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=False)

    cu.load_test_checkpoint(cfg, model)

    # Create video loaders.
    video_loader = loader.construct_loader(cfg, "test")

    # Create saver
    saver = Saver(
        cfg.DATA.PATH_TO_DATA_DIR, video_loader.dataset
    )

    model.eval()
    for i, (inputs, index) in tqdm(enumerate(video_loader), total=len(video_loader)):
        for i in range(len(inputs)):
            inputs[i] = inputs[i].cuda(non_blocking=True)
        index = index.cuda()
        feats = model(inputs)

        # Gather all the predictions across all the devices to perform ensemble.
        if cfg.NUM_GPUS > 1:
            feats, index = du.all_gather([feats, index])

        saver.save(feats, index)

    saver.merge()


def launch_job(cfg, init_method, func, daemon=False):
    """
    Run 'func' on one or more GPUs, specified in cfg
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        init_method (str): initialization method to launch the job with multiple
            devices.
        func (function): job to run on GPU(s)
        daemon (bool): The spawned processes’ daemon flag. If set to True,
            daemonic processes will be created
    """
    if cfg.NUM_GPUS > 1:
        torch.multiprocessing.spawn(
            mpu.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                func,
                init_method,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                cfg.DIST_BACKEND,
                cfg,
            ),
            daemon=daemon,
        )
    else:
        func(cfg=cfg)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args)

    launch_job(cfg=cfg, init_method=args.init_method, func=inference)
