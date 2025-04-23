from torch import Tensor
import torch
from torch import nn
import torch.utils.checkpoint
import contextlib
import torchvision
from einops import rearrange
import math
from stgcn_layers import Graph, get_stgcn_chain
import torch.nn.functional as F

class UniSignGNNSkeletonExtractor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.modes = ['body', 'left', 'right', 'face_all']
            
        self.graph, A = {}, []
        # project (x,y,score) to hidden dim
        hidden_dim = args.hidden_dim
        self.proj_linear = nn.ModuleDict()
        for mode in self.modes:
            self.graph[mode] = Graph(layout=f'{mode}', strategy='distance', max_hop=1)
            A.append(torch.tensor(self.graph[mode].A, dtype=torch.float32, requires_grad=False))
            self.proj_linear[mode] = nn.Linear(3, 64)
        self.gcn_modules = nn.ModuleDict()
        self.fusion_gcn_modules = nn.ModuleDict()
        spatial_kernel_size = A[0].size(0)

        for index, mode in enumerate(self.modes):
            self.gcn_modules[mode], final_dim = get_stgcn_chain(64, 'spatial', (1, spatial_kernel_size), A[index].clone(), True)
            self.fusion_gcn_modules[mode], _ = get_stgcn_chain(final_dim, 'temporal', (5, spatial_kernel_size), A[index].clone(), True)
        
        self.gcn_modules['left'] = self.gcn_modules['right']
        self.fusion_gcn_modules['left'] = self.fusion_gcn_modules['right']
        self.proj_linear['left'] = self.proj_linear['right']
        self.part_para = nn.Parameter(torch.zeros(hidden_dim*len(self.modes)))
        self.pose_proj = nn.Linear(256*4, 768)

    def forward(self, src_input, T_target = None):
        features = []

        body_feat = None
        for part in self.modes:
            # project position to hidden dim
            proj_feat = self.proj_linear[part](src_input[part]).permute(0,3,1,2) #B,C,T,V
            # spatial gcn forward
            gcn_feat = self.gcn_modules[part](proj_feat)
            if part == 'body':
                body_feat = gcn_feat

            else:
                assert not body_feat is None
                if part == 'left':
                    # Pose RGB fusion
                    gcn_feat = gcn_feat + body_feat[..., -2][...,None].detach()
                    
                elif part == 'right':
                    # Pose RGB fusion
                    gcn_feat = gcn_feat + body_feat[..., -1][...,None].detach()

                elif part == 'face_all':
                    gcn_feat = gcn_feat + body_feat[..., 0][...,None].detach()

                else:
                    raise NotImplementedError
            
            # temporal gcn forward
            gcn_feat = self.fusion_gcn_modules[part](gcn_feat) #B,C,T,V


            if T_target != None:
        
                B, C, T, V = gcn_feat.shape
                # Pool over time
                gcn_feat = gcn_feat.permute(0, 3, 1, 2).reshape(B*V, C, T)  # (B*V, C, T)
                gcn_feat = F.adaptive_avg_pool1d(gcn_feat, output_size=T_target)  # (B*V, C, T')
                gcn_feat = gcn_feat.reshape(B, V, C, T_target).permute(0, 2, 3, 1) #(, *V, C, T')

            pool_feat = gcn_feat.mean(-1).transpose(1,2) #B,T,C
            features.append(pool_feat) 

        # concat sub-pose feature across token dimension
        inputs_embeds = torch.cat(features, dim=-1) + self.part_para
        inputs_embeds = self.pose_proj(inputs_embeds) #(1, 225, 768)
        return inputs_embeds