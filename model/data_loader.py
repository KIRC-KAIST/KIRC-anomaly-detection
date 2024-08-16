import os
import pickle


import torch
import cv2
from torch.utils.data import Dataset
from torchvision.datasets.video_utils import VideoClips


# Return the cached VideoCilp objecet (pickle)
class VideoDataset(Dataset):
    
    def __init__(self, video_clip_path, dataframe, frame_diff=None, video_transform=None):
        self.video_clip_path = video_clip_path
        self.df = dataframe
        self.video_transform = video_transform
        self.frame_diff = frame_diff
        
        with open(str(self.video_clip_path), 'rb') as f:
            self.video_clips = pickle.load(f)
        
    def getitem_from_raw_video(self, index):
        video, _, _, _ = self.video_clips.get_clip(index)
        video_idx, clip_idx = self.video_clips.get_clip_location(index)
        video_path = self.video_clips.video_paths[video_idx]
        label = self._get_label(video_path)
        
        
        if self.video_transform:            
            video = video.numpy()
            video = self.video_transform(video)
            
            
        return video, label, video_path
    
    def _get_label(self, video_path):
        label = self.df[self.df['video_path']==video_path]['label'].unique().item()
        
        if (label == 0) or (label ==0.):
            return 0
        else:
            return 1        
        
    def __getitem__(self, index):
        video, label, video_path = self.getitem_from_raw_video(index)
        return video, label, video_path  # video: (N, T, H, W, C)
    

    def __len__(self):
        return len(self.df)

    