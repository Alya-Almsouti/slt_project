import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import os
import random
import numpy as np
import copy
import pickle
from decord import VideoReader, cpu
import json
import pathlib
from torchvision import transforms
# import importlib
# import config
# importlib.reload(config)
from config import rgb_dirs, pose_dirs
import torchvision.transforms as transforms
import pandas as pd
import cv2

# load sub-pose
def load_part_kp(skeletons, confs, force_ok=False):
    thr = 0.3
    kps_with_scores = {}
    scale = None
    
    for part in ['body', 'left', 'right', 'face_all']:
        kps = []
        confidences = []
        
        for skeleton, conf in zip(skeletons, confs):
            skeleton = skeleton[0]
            conf = conf[0]
            
            if part == 'body':
                hand_kp2d = skeleton[[0] + [i for i in range(3, 11)], :]
                confidence = conf[[0] + [i for i in range(3, 11)]]
            elif part == 'left':
                hand_kp2d = skeleton[91:112, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[91:112]
            elif part == 'right':
                hand_kp2d = skeleton[112:133, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[112:133]
            elif part == 'face_all':
                hand_kp2d = skeleton[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53], :]
                hand_kp2d = hand_kp2d - hand_kp2d[-1, :]
                confidence = conf[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53]]

            else:
                raise NotImplementedError
            
            kps.append(hand_kp2d)
            confidences.append(confidence)
            
        kps = np.stack(kps, axis=0)
        confidences = np.stack(confidences, axis=0)
        
        if part == 'body':
            if force_ok:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[...,None]], axis=-1), thr)

            else:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[...,None]], axis=-1), thr)
        else:
            assert not scale is None
            result = np.concatenate([kps, confidences[...,None]], axis=-1)
            if scale==0:
                result = np.zeros(result.shape)
            else:
                result[...,:2] = (result[..., :2]) / scale
                result = np.clip(result, -1, 1)
                # mask useless kp
                result[result[...,2]<=thr] = 0
            
        kps_with_scores[part] = torch.tensor(result)
        
    return kps_with_scores


# input: T, N, 3
# input is un-normed joints
def crop_scale(motion, thr):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]>thr][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    # ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    ratio = 1
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2] - [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    # mask useless kp
    result[result[...,2]<=thr] = 0
    return result, scale, [xs,ys]


# bbox of hands
def bbox_4hands(left_keypoints, right_keypoints, hw):
    # keypoints --> T,21,2
    # keypoints --> T,21,2
    
    def compute_bbox(keypoints):
        min_x = np.min(keypoints[..., 0], axis=1)
        min_y = np.min(keypoints[..., 1], axis=1)
        max_x = np.max(keypoints[..., 0], axis=1)
        max_y = np.max(keypoints[..., 1], axis=1)
        
        return (max_x+min_x)/2, (max_y+min_y)/2, (max_x-min_x), (max_y-min_y)
    H,W = hw
    
    if left_keypoints is None:
        left_keypoints = np.zeros([1,21,2])
        
    if right_keypoints is None:
        right_keypoints = np.zeros([1,21,2])
    # [T, 21, 2]
    left_mean_x, left_mean_y, left_diff_x, left_diff_y = compute_bbox(left_keypoints)
    left_mean_x = W*left_mean_x
    left_mean_y = H*left_mean_y
    
    left_diff_x = W*left_diff_x
    left_diff_y = H*left_diff_y
    
    left_diff_x = max(left_diff_x)
    left_diff_y = max(left_diff_y)
    left_box_hw = max(left_diff_x,left_diff_y)
    
    right_mean_x, right_mean_y, right_diff_x, right_diff_y = compute_bbox(right_keypoints)
    right_mean_x = W*right_mean_x
    right_mean_y = H*right_mean_y
    
    right_diff_x = W*right_diff_x
    right_diff_y = H*right_diff_y
    
    right_diff_x = max(right_diff_x)
    right_diff_y = max(right_diff_y)
    right_box_hw = max(right_diff_x,right_diff_y)
    
    box_hw = int(max(left_box_hw, right_box_hw) * 1.2 / 2) * 2
    box_hw = max(box_hw, 0)

    left_new_box = np.stack([left_mean_x - box_hw/2, left_mean_y - box_hw/2, left_mean_x + box_hw/2, left_mean_y + box_hw/2]).astype(np.int16)
    right_new_box = np.stack([right_mean_x - box_hw/2, right_mean_y - box_hw/2, right_mean_x + box_hw/2, right_mean_y + box_hw/2]).astype(np.int16)
    
    return left_new_box.transpose(1,0), right_new_box.transpose(1,0), box_hw

def load_support_rgb_dict(tmp, skeletons, confs, full_path, data_transform):
    support_rgb_dict = {}
    
    confs = np.array(confs)
    skeletons = np.array(skeletons) 

    # sample index of low scores
    left_confs_filter = confs[:,0,91:112].mean(-1)
    left_confs_filter_indices = np.where(left_confs_filter > 0.3)[0]

    if len(left_confs_filter_indices) == 0:
        left_sampled_indices = None
        left_skeletons = None
    else:
        
        left_confs = confs[left_confs_filter_indices]
        left_confs = left_confs[:,0,[95,99,103,107,111]].min(-1)
        
        left_weights = np.max(left_confs) - left_confs + 1e-5
        left_probabilities = left_weights / np.sum(left_weights)
        
        left_sample_size = int(np.ceil(0.1 * len(left_confs_filter_indices)))
        
        left_sampled_indices = np.random.choice(left_confs_filter_indices.tolist(), 
                                                size=left_sample_size, 
                                                replace=False, 
                                                p=left_probabilities)
        # left_sampled_indices: values: 0-255(0,max_len)
        # tmp: values: 0-(end-start)
        left_sampled_indices = np.sort(left_sampled_indices)
        
        left_skeletons = skeletons[left_sampled_indices,0,91:112]

    right_confs_filter = confs[:,0,112:].mean(-1)
    right_confs_filter_indices = np.where(right_confs_filter > 0.3)[0]
    if len(right_confs_filter_indices) == 0:
        right_sampled_indices = None
        right_skeletons = None
        
    else:
        right_confs = confs[right_confs_filter_indices]
        right_confs = right_confs[:,0,[95+21,99+21,103+21,107+21,111+21]].min(-1)

        right_weights = np.max(right_confs) - right_confs + 1e-5
        right_probabilities = right_weights / np.sum(right_weights)
        
        right_sample_size = int(np.ceil(0.1 * len(right_confs_filter_indices)))
        
        right_sampled_indices = np.random.choice(right_confs_filter_indices.tolist(), 
                                                 size=right_sample_size, 
                                                 replace=False, 
                                                 p=right_probabilities)
        right_sampled_indices = np.sort(right_sampled_indices)
        
        right_skeletons = skeletons[right_sampled_indices,0,112:133]
        
    image_size = 112
    all_indices = []
    if not left_sampled_indices is None:
        all_indices.append(left_sampled_indices)
    if not right_sampled_indices is None:
        all_indices.append(right_sampled_indices)
    if len(all_indices) == 0:
        support_rgb_dict['left_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['left_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['left_skeletons_norm'] = torch.zeros(1, 21, 2)
        
        support_rgb_dict['right_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['right_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['right_skeletons_norm'] = torch.zeros(1, 21, 2)

        return support_rgb_dict

    sampled_indices = np.concatenate(all_indices)
    sampled_indices = np.unique(sampled_indices)
    sampled_indices_real = tmp[sampled_indices]

    # load image sample
    imgs = load_video_support_rgb(full_path, sampled_indices_real)

    # get hand bbox
    left_new_box, right_new_box, box_hw = bbox_4hands(left_skeletons,
                                                        right_skeletons,
                                                        imgs[0].shape[:2])
    
    # crop left and right hand
    image_size = 112
    if box_hw == 0:
        support_rgb_dict['left_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['left_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['left_skeletons_norm'] = torch.zeros(1, 21, 2)
        
        support_rgb_dict['right_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['right_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['right_skeletons_norm'] = torch.zeros(1, 21, 2)

        return support_rgb_dict

    factor = image_size / box_hw
    
    if left_sampled_indices is None:
        left_hands = torch.zeros(1, 3, image_size, image_size)
        left_skeletons_norm = torch.zeros(1, 21, 2)
        
    else:
        left_hands = torch.zeros(len(left_sampled_indices), 3, image_size, image_size)
            
        left_skeletons_norm = left_skeletons * imgs[0].shape[:2][::-1] - left_new_box[:, None, [0,1]]
        left_skeletons_norm = left_skeletons_norm / box_hw
        left_skeletons_norm = left_skeletons_norm.clip(0,1)

    if right_sampled_indices is None:
        right_hands = torch.zeros(1, 3, image_size, image_size)
        right_skeletons_norm = torch.zeros(1, 21, 2)
        
    else:
        right_hands = torch.zeros(len(right_sampled_indices), 3, image_size, image_size)
        
        right_skeletons_norm = right_skeletons * imgs[0].shape[:2][::-1] - right_new_box[:, None, [0,1]]
        right_skeletons_norm = right_skeletons_norm / box_hw
        right_skeletons_norm = right_skeletons_norm.clip(0,1)
    left_idx = 0
    right_idx = 0

    for idx, img in enumerate(imgs):
        mapping_idx = sampled_indices[idx]
        if not left_sampled_indices is None and left_idx < len(left_sampled_indices) and mapping_idx == left_sampled_indices[left_idx]:
            box = left_new_box[left_idx]
            
            img_draw = np.uint8(copy.deepcopy(img))[box[1]:box[3],box[0]:box[2],:]
            img_draw = np.pad(img_draw, ((0, max(0, box_hw-img_draw.shape[0])), (0, max(0, box_hw-img_draw.shape[1])), (0, 0)), mode='constant', constant_values=0)
            
            f_img = Image.fromarray(img_draw).convert('RGB').resize((image_size, image_size))
            f_img = data_transform(f_img).unsqueeze(0)
            left_hands[left_idx] = f_img
            left_idx += 1
            
        if not right_sampled_indices is None and right_idx < len(right_sampled_indices) and mapping_idx == right_sampled_indices[right_idx]:
            box = right_new_box[right_idx]
            
            img_draw = np.uint8(copy.deepcopy(img))[box[1]:box[3],box[0]:box[2],:]
            img_draw = np.pad(img_draw, ((0, max(0, box_hw-img_draw.shape[0])), (0, max(0, box_hw-img_draw.shape[1])), (0, 0)), mode='constant', constant_values=0)
            
            f_img = Image.fromarray(img_draw).convert('RGB').resize((image_size, image_size))
            f_img = data_transform(f_img).unsqueeze(0)
            right_hands[right_idx] = f_img
            right_idx += 1
   
    if left_sampled_indices is None:
        left_sampled_indices = np.array([-1])
        
    if right_sampled_indices is None:
        right_sampled_indices = np.array([-1])

    # get index, images and keypoints priors
    support_rgb_dict['left_sampled_indices'] = torch.tensor(left_sampled_indices) #number of frames sampled from the keypoints
    support_rgb_dict['left_hands'] = left_hands
    support_rgb_dict['left_skeletons_norm'] = torch.tensor(left_skeletons_norm)
    
    support_rgb_dict['right_sampled_indices'] = torch.tensor(right_sampled_indices)
    support_rgb_dict['right_hands'] = right_hands
    support_rgb_dict['right_skeletons_norm'] = torch.tensor(right_skeletons_norm)

    return support_rgb_dict


# use split rgb video for save time
def load_video_support_rgb(path, transform):
    # vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    
    # vr.seek(0)
    # buffer = vr.get_batch(tmp).asnumpy()
    # batch_image = buffer
    # del vr

    # return batch_image
    cap = cv2.VideoCapture(path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color format
        frame = Image.fromarray(frame)
        frame = transform(frame)  # Apply transformation  # Convert to tensor
        frames.append(frame)
    
    cap.release()
    return torch.stack(frames)

# build base dataset
class Base_Dataset(Dataset.Dataset):
    def collate_fn(self, batch):
        tgt_batch,src_length_batch,name_batch,pose_tmp,gloss_batch, frames_batch = [],[],[],[],[],[]
        
        for name_sample, pose_sample, text, gloss, frames in batch:
            name_batch.append(name_sample)
            pose_tmp.append(pose_sample)
            tgt_batch.append(text)
            gloss_batch.append(gloss)
            frames_batch.append(frames)
            

        src_input = {}

        keys = pose_tmp[0].keys()
        max_len = -1
        for key in keys:
            max_len = max([len(vid[key]) for vid in pose_tmp])
            video_length = torch.LongTensor([len(vid[key]) for vid in pose_tmp])
            
            padded_video = [torch.cat(
                (
                    vid[key],
                    vid[key][-1][None].expand(max_len - len(vid[key]), -1, -1),
                )
                , dim=0)
                for vid in pose_tmp]
            
            img_batch = torch.stack(padded_video,0)
            
            src_input[key] = img_batch
            if 'attention_mask' not in src_input.keys():
                src_length_batch = video_length

                mask_gen = []
                for i in src_length_batch:
                    tmp = torch.ones([i]) + 7
                    mask_gen.append(tmp)
                mask_gen = pad_sequence(mask_gen, padding_value=0,batch_first=True)
                img_padding_mask = (mask_gen != 0).long()
                src_input['attention_mask'] = img_padding_mask

                src_input['name_batch'] = name_batch
                src_input['src_length_batch'] = src_length_batch
        
        if self.rgb_support:
            padded_video = []
            for frames in frames_batch:
                T, C, H, W = frames.shape
                pad_len = max_len - T
                if pad_len > 0:
                    # Repeat last frame
                    pad = frames[-1].unsqueeze(0).expand(pad_len, C, H, W)
                    padded = torch.cat([frames, pad], dim=0)
                else:
                    padded = frames[:max_len]  # Optional: trim if too long
                padded_video.append(padded)
            img_batch = torch.stack(padded_video,0)
            src_input['frames'] = img_batch
        tgt_input = {}
        tgt_input['gt_sentence'] = tgt_batch
        tgt_input['gt_gloss'] = gloss_batch

        return src_input, tgt_input


class S2T_Dataset_news(Base_Dataset):
    def __init__(self, path, args, phase):
        super(S2T_Dataset_news, self).__init__()
        self.args = args
        self.rgb_support = self.args.rgb_support
        self.phase = phase
        self.max_length = args.max_length

        path = pathlib.Path(path)
        print('dataset path::: ')
        print(path)
        print(self.max_length)

        with path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)
        if self.args.dataset == "CSL_News" :
            self.pose_dir = pose_dirs[args.dataset]
            self.rgb_dir = rgb_dirs[args.dataset]
        else:
            raise NotImplementedError
        sum_sample = len(self.annotation)
        print('dataset length:: ', sum_sample)
        self.new_size = 224
        self.data_transform = transforms.Compose([
                                    transforms.Resize((self.new_size, self.new_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])

        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        num_retries = 10  

        # skip some invalid video sample
        for _ in range(num_retries):
            sample = self.annotation[index]

            text = sample['text']
            name_sample = sample['video']
           
            try:
                pose_sample, support_rgb_dict = self.load_pose(sample['pose'], sample['video'])
            except:
                import traceback

                traceback.print_exc()
                print(f"Failed to load examples with video: {name_sample}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue

            break
           
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        
        return name_sample, pose_sample, text, _, support_rgb_dict
    
    def load_pose(self, pose_name, rgb_name):
        pose = pickle.load(open(os.path.join(self.pose_dir, pose_name), 'rb'))
        full_path = os.path.join(self.rgb_dir, rgb_name)
        
        duration = len(pose['scores'])

        if duration > self.max_length:
            print('Insisde this')
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))
        
        tmp = np.array(tmp)
            
        # dict_keys(['keypoints', 'scores'])
        # keypoints (1, 133, 2)
        # scores (1, 133)
        
        skeletons = pose['keypoints']
        confs = pose['scores']
        confs = np.expand_dims(confs, axis=1)
        skeletons = np.expand_dims(skeletons, axis=1)
        skeletons_tmp = []
        confs_tmp = []
        
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp
                
        kps_with_scores = load_part_kp(skeletons, confs)
        
        support_rgb_dict = {}
        if self.rgb_support:
            support_rgb_dict = load_video_support_rgb(full_path, self.data_transform)

        return kps_with_scores, support_rgb_dict

    def __str__(self):
        return f'#total {len(self)}'


class VidText_Dataset(Base_Dataset):
    def __init__(self, path, args, transform = None):
        super(VidText_Dataset, self).__init__()
        self.args = args
        self.max_length = args.max_length
        self.annotations = pd.read_table(path, low_memory=False)
        print(self.annotations.columns)
        self.transform = transform
        self.new_size = 224
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((self.new_size, self.new_size)),  # Resize frames to 128x128
                transforms.ToTensor()
            ])
        
        if self.args.dataset in ["Open_ASL"]:
            self.video_dir = rgb_dirs[args.dataset]
        else:
            raise NotImplementedError("Dataset not supported")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        name_sample = self.annotations.iloc[idx]['video_name']
        text = self.annotations.iloc[idx]['caption']
        
        # Load video frames
        frames = self.load_video(f"{name_sample}.mp4")
        
        return name_sample, frames, text
    
    def load_video(self, path):
        full_path = os.path.join(self.video_dir, path)
        cap = cv2.VideoCapture(full_path)
        frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color format
            frame = Image.fromarray(frame)
            frame = self.transform(frame)  # Apply transformation  # Convert to tensor
            frames.append(frame)
        
        cap.release()
        
        # Handle cases where frames exceed or fall short of max_length
        num_frames = len(frames)
        if num_frames > self.max_length:
            selected_indices = sorted(random.sample(range(num_frames), k=self.max_length))
            frames = [frames[i] for i in selected_indices]
        elif num_frames < self.max_length:
            last_frame = frames[-1] if frames else torch.zeros((3, self.new_size, self.new_size))  # Default blank frame if empty
            frames.extend([last_frame] * (self.max_length - num_frames))
        
        return torch.stack(frames)
    
    def __str__(self):
        return f'#total {len(self)}'
    def collate_fn(self, batch):
        name_batch, frames_batch, text_batch = [], [], []
        
        for name_sample, frames, text in batch:
            name_batch.append(name_sample)
            frames_batch.append(frames)
            text_batch.append(text)
        
        src_input = {}
        
        max_len = max(len(frames) for frames in frames_batch)
        video_length = torch.tensor([len(frames) for frames in frames_batch], dtype=torch.long)
        
        padded_video = [
            torch.cat(
                (frames, frames[-1].unsqueeze(0).expand(max_len - len(frames), -1, -1, -1)),
                dim=0
            ) if len(frames) < max_len else frames
            for frames in frames_batch
        ]
        
        img_batch = torch.stack(padded_video, dim=0)
        src_input['video'] = img_batch
        src_input['name_batch'] = name_batch
        src_input['src_length_batch'] = video_length
        
        tgt_input = {
            'gt_sentence': text_batch
        }
        
        return src_input, tgt_input

