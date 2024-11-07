import torch
from torch import nn


class GGCA(nn.Module):  #(Global Grouped Coordinate Attention) 全局分组坐标注意力
    def __init__(self, channel, h, w, reduction=16, num_groups=5):
        super(GGCA, self).__init__()
        self.num_groups = num_groups  # 分组数
        self.group_channels = channel // num_groups  # 每组的通道数
        self.h = h  # 高度方向的特定尺寸
        self.w = w  # 宽度方向的特定尺寸

        # 定义H方向的全局平均池化和最大池化
        self.avg_pool_h = nn.AdaptiveAvgPool2d((h, 1))  # 输出大小为(h, 1)
        self.max_pool_h = nn.AdaptiveMaxPool2d((h, 1))
        # 定义W方向的全局平均池化和最大池化
        self.avg_pool_w = nn.AdaptiveAvgPool2d((1, w))  # 输出大小为(1, w)
        self.max_pool_w = nn.AdaptiveMaxPool2d((1, w))

        # 定义共享的卷积层，用于通道间的降维和恢复
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.group_channels, out_channels=self.group_channels // reduction,
                      kernel_size=(1, 1)),
            nn.BatchNorm2d(self.group_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.group_channels // reduction, out_channels=self.group_channels,
                      kernel_size=(1, 1))
        )
        # 定义sigmoid激活函数
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        batch_size, channel, height, width = x.size()
        # 确保通道数可以被分组数整除,一般分组数,要选择整数,不然不能被整除。而且是小一点.groups选择4挺好。
        assert channel % self.num_groups == 0, "The number of channels must be divisible by the number of groups."

        # 将输入特征图按通道数分组
        x = x.view(batch_size, self.num_groups, self.group_channels, height, width)

        # 分别在H方向进行全局平均池化和最大池化
        x_h_avg = self.avg_pool_h(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, self.h, 1)
        x_h_max = self.max_pool_h(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, self.h, 1)

        # 分别在W方向进行全局平均池化和最大池化
        x_w_avg = self.avg_pool_w(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, 1, self.w)
        x_w_max = self.max_pool_w(x.view(batch_size * self.num_groups, self.group_channels, height, width)).view(
            batch_size, self.num_groups, self.group_channels, 1, self.w)

        # 应用共享卷积层进行特征处理
        y_h_avg = self.shared_conv(x_h_avg.view(batch_size * self.num_groups, self.group_channels, self.h, 1))
        y_h_max = self.shared_conv(x_h_max.view(batch_size * self.num_groups, self.group_channels, self.h, 1))

        y_w_avg = self.shared_conv(x_w_avg.view(batch_size * self.num_groups, self.group_channels, 1, self.w))
        y_w_max = self.shared_conv(x_w_max.view(batch_size * self.num_groups, self.group_channels, 1, self.w))

        # 计算注意力权重
        att_h = self.sigmoid_h(y_h_avg + y_h_max).view(batch_size, self.num_groups, self.group_channels, self.h, 1)
        att_w = self.sigmoid_w(y_w_avg + y_w_max).view(batch_size, self.num_groups, self.group_channels, 1, self.w)

        # 应用注意力权重
        out = x * att_h * att_w
        out = out.view(batch_size, channel, height, width)

        return out





import torch
import torch.nn as nn

from transformer_encoder import Encoder
from transformer_decoder import Decoder


class TrajectoryModel(nn.Module):

    def __init__(self, in_size, obs_len, pred_len, embed_size, enc_num_layers, int_num_layers_list, heads, forward_expansion):
        super(TrajectoryModel, self).__init__()

        self.embedding = nn.Linear(in_size*(obs_len + pred_len), embed_size)

        self.mode_encoder = Encoder(embed_size, enc_num_layers, heads, forward_expansion, islinear=True)
        self.cls_head = nn.Linear(embed_size, 1)

        self.nei_embedding = nn.Linear(in_size*obs_len, embed_size)
        self.social_decoder =  Decoder(embed_size, int_num_layers_list[1], heads, forward_expansion, islinear=False)
        self.reg_head = nn.Linear(embed_size, in_size*pred_len)

        self.ggca = GGCA(channel=100,h=20,w=2)

    def spatial_interaction(self, ped, neis, mask):
        
        # ped [B K embed_size]
        # neis [B N obs_len 2]  N is the max number of agents of current scene
        # mask [B N N] is used to stop the attention from invalid agents

        neis = neis.reshape(neis.shape[0], neis.shape[1], -1)  # [B N obs_len*2]
        nei_embeddings = self.nei_embedding(neis)  # [B N embed_size]
        
        mask = mask[:, 0:1].repeat(1, ped.shape[1], 1)  # [B K N]
        int_feat = self.social_decoder(ped, nei_embeddings, mask)  # [B K embed_size]

        return int_feat # [B K embed_size]
    
    def forward(self, ped_obs, neis_obs, motion_modes, mask, closest_mode_indices, test=False, num_k=20):

        # ped_obs [B obs_len 2]
        # nei_obs [B N obs_len 2]
        # motion_modes [K pred_len 2]
        # closest_mode_indices [B]

        ped_obs = ped_obs.unsqueeze(1).repeat(1, motion_modes.shape[0], 1, 1)  # [B K obs_len 2]
        motion_modes = motion_modes.unsqueeze(0).repeat(ped_obs.shape[0], 1, 1, 1)

        ped_seq = torch.cat((ped_obs, motion_modes), dim=-2)  # [B K seq_len 2] seq_len = obs_len + pred_len
        ped_seq = self.ggca(ped_seq)
        ped_seq = ped_seq.reshape(ped_seq.shape[0], ped_seq.shape[1], -1)  # [B K seq_len*2]
        ped_embedding = self.embedding(ped_seq) # [B K embed_size]
        
        ped_feat = self.mode_encoder(ped_embedding)  # [B K embed_size]
        scores = self.cls_head(ped_feat).squeeze()  # [B K]

        if not test:
            index1 = torch.LongTensor(range(closest_mode_indices.shape[0])).cuda()  # [B]
            index2 = closest_mode_indices
            closest_feat = ped_feat[index1, index2].unsqueeze(1)  # [B 1 embed_size]

            int_feat = self.spatial_interaction(closest_feat, neis_obs, mask)  # [B 1 embed_size]
            pred_traj = self.reg_head(int_feat.squeeze())  # [B pred_len*2]

            return pred_traj, scores

        if test:
            top_k_indices = torch.topk(scores, k=num_k, dim=-1).indices  # [B num_k]
            top_k_indices = top_k_indices.flatten()  # [B*num_k]
            index1 = torch.LongTensor(range(ped_feat.shape[0])).cuda()  # [B]
            index1 = index1.unsqueeze(1).repeat(1, num_k).flatten() # [B*num_k]
            index2 = top_k_indices # [B*num_k]
            top_k_feat = ped_feat[index1, index2]  # [B*num_k embed_size]
            top_k_feat = top_k_feat.reshape(ped_feat.shape[0], num_k, -1)  # [B num_k embed_size]

            int_feats = self.spatial_interaction(top_k_feat, neis_obs, mask)  # [B num_k embed_size]
            pred_trajs = self.reg_head(int_feats)  # [B num_k pred_size*2]

            return pred_trajs, scores