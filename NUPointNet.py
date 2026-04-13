"""
NUPoint-Net
"""
from re import I
from typing import List, Type
import logging
import torch
import torch.nn as nn
from ..build import MODELS
from ..layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \
    create_grouper, create_grouper_idx, furthest_point_sample, random_sample, three_interpolation
import copy
import math

class GaussianKDEOffsetGenerator(nn.Module):
    """
    Input:
        coords: [BN, K, 3]
    Output:
        delta:  [BN, K, 3]
    """
    def __init__(self, pe_dim=32, hidden_dim=16, sigma=1.0, eps=1e-6):
        super().__init__()
        self.sigma = sigma
        self.eps = eps

        # M1 in the paper: PE(p_j - p_j^k)
        self.pe_mlp = nn.Sequential(
            nn.Linear(3, pe_dim),
            nn.ReLU(inplace=True)
        )

        # M2, M3 in the paper: FFN(PE * GKDE)
        self.ffn = nn.Sequential(
            nn.Linear(pe_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, coords):
        """
        coords: [BN, K, 3]
        """
        BN, K, _ = coords.shape
        rel = coords.unsqueeze(2) - coords.unsqueeze(1)

        # -----------------------------
        # 1) Position Embedding (PE)
        # -----------------------------
        pe = self.pe_mlp(rel)
        pe = pe.mean(dim=2)

        # -----------------------------
        # 2) Gaussian KDE
        # -----------------------------
        dist2 = (rel ** 2).sum(dim=-1)  # [BN, K, K]
        coef = 1.0 / math.sqrt(2.0 * math.pi * (self.sigma ** 2) + self.eps)
        kde = coef * torch.exp(-dist2 / (2.0 * (self.sigma ** 2) + self.eps))
        kde = kde.sum(dim=2, keepdim=True)  # [BN, K, 1]
        kde = kde / (kde.amax(dim=1, keepdim=True) + self.eps)

        # -----------------------------
        # 3) Density-adaptive offset
        # -----------------------------
        fused = pe * kde
        delta = self.ffn(fused)

        return delta

def get_reduction_fn(reduction):
    reduction = 'mean' if reduction.lower() == 'avg' else reduction
    assert reduction in ['sum', 'max', 'mean']
    if reduction == 'max':
        pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
    elif reduction == 'mean':
        pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
    elif reduction == 'sum':
        pool = lambda x: torch.sum(x, dim=-1, keepdim=False)
    return pool


def get_aggregation_feautres(p, dp, f, fj, feature_type='dp_fj'):
    if feature_type == 'dp_fj':
        fj = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, fj, df], 1)
    elif feature_type == 'pi_dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1]), dp, fj, df], 1)
    elif feature_type == 'dp_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, df], 1)
    return fj


def group_by_idx(query_xyz, support_xyz, support_features, idx):
    B, Nq, K = idx.shape
    grouped_xyz = torch.gather(
        support_xyz.unsqueeze(1).expand(-1, Nq, -1, -1), 2,
        idx.unsqueeze(-1).expand(-1, -1, -1, support_xyz.shape[-1]))
    dp = (grouped_xyz - query_xyz.unsqueeze(2)).permute(0, 3, 1, 2).contiguous()

    fj = None
    if support_features is not None:
        grouped_f = torch.gather(
            support_features.transpose(1, 2).unsqueeze(1).expand(-1, Nq, -1, -1), 2,
            idx.unsqueeze(-1).expand(-1, -1, -1, support_features.shape[1]))
        fj = grouped_f.permute(0, 3, 1, 2).contiguous()
    return dp, fj


class LocalAggregation(nn.Module):
    """Local aggregation layer for a set 
    Set abstraction layer abstracts features from a larger set to a smaller set
    Local aggregation layer aggregates features from the same set
    """

    def __init__(self,
                 channels: List[int],
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 conv_args=None,
                 feature_type='dp_fj',
                 reduction='max',
                 last_act=True,
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        channels1 = channels 
        convs1 = []
        for i in range(len(channels1) - 1):  # #layers in each blocks
            convs1.append(create_convblock1d(channels1[i], channels1[i + 1],
                                             norm_args=norm_args,
                                            act_args=None if i == (
                                                    len(channels1) - 2) and not last_act else act_args,
                                             **conv_args)
                          )
        self.convs1 = nn.Sequential(*convs1)
        self.grouper = create_grouper(group_args)
        self.reduction = reduction.lower()
        self.pool = get_reduction_fn(self.reduction)
        self.feature_type = feature_type

    def forward(self, pf, pe, knn_idx=None):
        # p: position, f: feature
        p, f = pf
        # preconv
        f = self.convs1(f)
        # grouping index cache
        if knn_idx is None:
            knn_idx = self.grouper(p, p, f)
        dp, fj = group_by_idx(p, p, f, knn_idx)
        # pe + fj 
        f = pe + fj
        f = self.pool(f)
        """ DEBUG neighbor numbers. 
        if f.shape[-1] != 1:
            query_xyz, support_xyz = p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(
                f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
        DEBUG end """
        return f, knn_idx


class SetAbstraction(nn.Module):
    """The modified set abstraction module in PointNet++ with residual connection support
    """

    def __init__(self,
                 in_channels, out_channels,
                 layers=1,
                 stride=1,
                 group_args={'NAME': 'ballquery',
                             'radius': 0.1, 'nsample': 16},
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 conv_args=None,
                 sampler='fps',
                 feature_type='dp_fj',
                 use_res=False,
                 is_head=False,
                 **kwargs, 
                 ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        self.feature_type = feature_type
        main_group_args = copy.deepcopy(group_args)
        dilated_group_args = copy.deepcopy(group_args)
        assign_group_args = copy.deepcopy(group_args)
        self.dilation_rate = kwargs.get('dilation_rate', 2)

        if self.all_aggr:
            main_group_args.nsample = None
            main_group_args.radius = None
            dilated_group_args.nsample = None
            dilated_group_args.radius = None
            assign_group_args.nsample = 1
            assign_group_args.radius = None
            self.k = None
        else:
            self.k = main_group_args.nsample
            dilated_group_args.nsample = int(self.k * self.dilation_rate)
            assign_group_args.nsample = 1

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * \
                   (layers - 1) + [out_channels]
        channels[0] = in_channels #if is_head else CHANNEL_MAP[feature_type](channels[0])
        channels1 = channels
        # channels2 = copy.copy(channels)
        channels2 = [in_channels] + [32,32] * (min(layers, 2) - 1) + [out_channels] # 16
        channels2[0] = 3
        convs1 = []
        convs2 = []

        self.deformable = GaussianKDEOffsetGenerator(pe_dim=32, hidden_dim=16, sigma=1.0)

        weight = []
        weight.append(create_convblock2d(channels1[1], channels1[1],
                                        norm_args=norm_args if not is_head else None,
                                        act_args=None, **conv_args))
        self.weight = nn.Sequential(*weight)

        if self.use_res:
            self.skipconv = create_convblock1d(
                in_channels, channels[-1], norm_args=None, act_args=None) if in_channels != channels[
                -1] else nn.Identity()
            self.act = create_act(act_args)

        # actually, one can use local aggregation layer to replace the following
        for i in range(len(channels1) - 1):  # #layers in each blocks
            convs1.append(create_convblock1d(channels1[i], channels1[i + 1],
                                             norm_args=norm_args if not is_head else None,
                                             act_args=None if i == len(channels) - 2
                                                            and (self.use_res or is_head) else act_args,
                                             **conv_args)
                          )
        self.convs1 = nn.Sequential(*convs1)

        if not is_head:
            for i in range(len(channels2) - 1):  # #layers in each blocks
                convs2.append(create_convblock2d(channels2[i], channels2[i + 1],
                                                 norm_args=norm_args if not is_head else None,
                                                #  act_args=None if i == len(channels) - 2
                                                #                 and (self.use_res or is_head) else act_args,
                                                 act_args=act_args,
                                                **conv_args)
                            )
            self.convs2 = nn.Sequential(*convs2)

            self.grouper = create_grouper_idx(main_group_args)
            self.dilated_grouper = create_grouper_idx(dilated_group_args)
            self.np_assign = create_grouper_idx(assign_group_args)

            # self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
            self.sads = False
            if sampler.lower() == 'fps':
                self.sample_fn = furthest_point_sample
            elif sampler.lower() == 'random':
                self.sample_fn = random_sample
            elif sampler.lower() == 'sads':
                self.sads = True
                self.sample_fn = furthest_point_sample

    def forward(self, pf_pe_idx):
        if len(pf_pe_idx) == 4:
            p, f, pe, sorted_idx = pf_pe_idx
        else:
            p, f, pe = pf_pe_idx
            sorted_idx = None
        B, N, _ = p.shape
        if self.is_head:
            f = self.convs1(f)  # (n, c)
        else:
            if not self.all_aggr:
                if self.sads:
                    ##### SADS #####
                    num_points = N // (2 * self.stride)
                    key_idx = sorted_idx[:, :num_points].long()

                    mask = torch.ones(B, N, dtype=torch.bool, device=p.device)
                    mask.scatter_(1, key_idx, False)
                    p_remain = p[mask].view(B, N - num_points, 3).contiguous()

                    fps_idx_local = self.sample_fn(p_remain, num_points).long()

                    all_idx = torch.arange(N, device=p.device).unsqueeze(0).expand(B, -1)
                    remain_idx = all_idx[mask].view(B, N - num_points)
                    fps_idx_global = torch.gather(remain_idx, 1, fps_idx_local)

                    idx = torch.cat([key_idx, fps_idx_global], dim=1).long()
                    new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3)).contiguous()
                else:
                    idx = self.sample_fn(p, p.shape[1] // self.stride).long()
                    new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3)).contiguous()
            else:
                new_p = p
            """ DEBUG neighbor numbers. 
            query_xyz, support_xyz = new_p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
            DEBUG end """
            if self.use_res or 'df' in self.feature_type:
                idx = self.sample_fn(p, p.shape[1] // self.stride).long()
                fi = torch.gather(
                    f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
                if self.use_res:
                    identity = self.skipconv(fi)
            else:
                fi = None
            
            ##### D2LA #####
            f = self.convs1(f)
            p = p[..., :3].contiguous()
            new_p = new_p[..., :3].contiguous()

            dilated_idx = self.dilated_grouper(new_p, p, f)
            dp_dilated, fj_dilated = group_by_idx(new_p, p, f, dilated_idx)
            B, c, Nq, Kd = fj_dilated.shape

            dp_dilated_xyz = dp_dilated.permute(0, 2, 3, 1).contiguous().view(B * Nq, Kd, 3)
            fj_dilated_bn = fj_dilated.permute(0, 2, 3, 1).contiguous().view(B * Nq, Kd, c)

            kernel_idx = self.sample_fn(dp_dilated_xyz, self.k).long()
            kernel_xyz = torch.gather(dp_dilated_xyz, 1, kernel_idx.unsqueeze(-1).expand(-1, -1, 3)).contiguous()
            kernel_fj = torch.gather(fj_dilated_bn, 1, kernel_idx.unsqueeze(-1).expand(-1, -1, c)).contiguous()

            delta = self.deformable(kernel_xyz)
            deformed_xyz = (kernel_xyz + delta).contiguous()

            support_f = kernel_fj.permute(0, 2, 1).contiguous()
            nn_idx = self.np_assign(deformed_xyz, kernel_xyz, support_f)
            nn_dp, nn_fj = group_by_idx(deformed_xyz, kernel_xyz, support_f, nn_idx)

            dp = nn_dp.view(B, Nq, 3, self.k).permute(0, 2, 1, 3).contiguous()
            fj = nn_fj.view(B, Nq, c, self.k).permute(0, 2, 1, 3).contiguous()
            modulation = torch.softmax(self.weight(fj), dim=-1)
            pe = self.convs2(dp)
            f = self.pool((pe + fj) * modulation)
            if self.use_res:
                f = self.act(f + identity)
            p = new_p
        return p, f, pe


class FeaturePropogation(nn.Module):
    """The Feature Propogation module in PointNet++
    """

    def __init__(self, mlp,
                 upsample=True,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'}
                 ):
        """
        Args:
            mlp: [current_channels, next_channels, next_channels]
            out_channels:
            norm_args:
            act_args:
        """
        super().__init__()
        if not upsample:
            self.linear2 = nn.Sequential(
                nn.Linear(mlp[0], mlp[1]), nn.ReLU(inplace=True))
            mlp[1] *= 2
            linear1 = []
            for i in range(1, len(mlp) - 1):
                linear1.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                  norm_args=norm_args, act_args=act_args
                                                  ))
            self.linear1 = nn.Sequential(*linear1)
        else:
            convs = []
            for i in range(len(mlp) - 1):
                convs.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                norm_args=norm_args, act_args=act_args
                                                ))
            self.convs = nn.Sequential(*convs)

        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, pf1, pf2=None):
        # pfb1 is with the same size of upsampled points
        if pf2 is None:
            _, f = pf1  # (B, N, 3), (B, C, N)
            f_global = self.pool(f)
            f = torch.cat(
                (f, self.linear2(f_global).unsqueeze(-1).expand(-1, -1, f.shape[-1])), dim=1)
            f = self.linear1(f)
        else:
            p1, f1 = pf1
            p2, f2 = pf2
            if f1 is not None:
                f = self.convs(
                    torch.cat((f1, three_interpolation(p1, p2, f2)), dim=1))
            else:
                f = self.convs(three_interpolation(p1, p2, f2))
        return f


class InvResMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 num_posconvs=2,#2,
                 less_act=False,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = int(in_channels * expansion)
        self.convs = LocalAggregation([in_channels, in_channels],
                                      norm_args=norm_args, act_args=act_args ,#if num_posconvs > 0 else None,
                                      group_args=group_args, conv_args=conv_args,
                                      **aggr_args, **kwargs)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        elif num_posconvs == 4:
            channels = [in_channels, in_channels, in_channels, in_channels, in_channels]
        elif num_posconvs == 3:
            channels = [in_channels, in_channels, in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = []
        # point wise after depth wise conv (without last layer)
        for i in range(len(channels) - 1):
            pwconv.append(create_convblock1d(channels[i], channels[i + 1],
                                             norm_args=norm_args,
                                             act_args=act_args if
                                             (i != len(channels) - 2) and not less_act else None,
                                             **conv_args)
                          )
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

    def forward(self, pf_pe):
        if len(pf_pe) == 4:
            p, f, pe, knn_idx = pf_pe
        else:
            p, f, pe = pf_pe
            knn_idx = None
        identity = f
        f, knn_idx = self.convs([p, f], pe, knn_idx)
        f = self.pwconv(f)
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        if len(pf_pe) == 4:
            return [p, f, pe, knn_idx]
        return [p, f, pe]


@MODELS.register_module()
class NUPointNetEncoder(nn.Module):
    r"""The Encoder for PointNext 
    `"PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies".
    <https://arxiv.org/abs/2206.04670>`_.
    .. note::
        For an example of using :obj:`PointNextEncoder`, see
        `examples/segmentation/main.py <https://github.com/guochengqian/PointNeXt/blob/master/cfgs/s3dis/README.md>`_.
    Args:
        in_channels (int, optional): input channels . Defaults to 4.
        width (int, optional): width of network, the output mlp of the stem MLP. Defaults to 32.
        blocks (List[int], optional): # of blocks per stage (including the SA block). Defaults to [1, 4, 7, 4, 4].
        strides (List[int], optional): the downsampling ratio of each stage. Defaults to [4, 4, 4, 4].
        block (strorType[InvResMLP], optional): the block to use for depth scaling. Defaults to 'InvResMLP'.
        nsample (intorList[int], optional): the number of neighbors to query for each block. Defaults to 32.
        radius (floatorList[float], optional): the initial radius. Defaults to 0.1.
        aggr_args (_type_, optional): the args for local aggregataion. Defaults to {'feature_type': 'dp_fj', "reduction": 'max'}.
        group_args (_type_, optional): the args for grouping. Defaults to {'NAME': 'ballquery'}.
        norm_args (_type_, optional): the args for normalization layer. Defaults to {'norm': 'bn'}.
        act_args (_type_, optional): the args for activation layer. Defaults to {'act': 'relu'}.
        expansion (int, optional): the expansion ratio of the InvResMLP block. Defaults to 4.
        sa_layers (int, optional): the number of MLP layers to use in the SA block. Defaults to 1.
        sa_use_res (bool, optional): wheter to use residual connection in SA block. Set to True only for PointNeXt-S. 
    """

    def __init__(self,
                 in_channels: int = 4,
                 width: int = 32,
                 blocks: List[int] = [1, 4, 7, 4, 4],
                 strides: List[int] = [4, 4, 4, 4],
                 block: str or Type[InvResMLP] = 'InvResMLP',
                 nsample: int or List[int] = 32,
                 radius: float or List[float] = 0.1,
                 aggr_args: dict = {'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args: dict = {'NAME': 'ballquery'},
                 sa_layers: int = 1,
                 sa_use_res: bool = False,
                 **kwargs
                 ):
        super().__init__()
        if isinstance(block, str):
            block = eval(block)
        self.blocks = blocks
        self.strides = strides
        self.in_channels = in_channels
        self.aggr_args = aggr_args
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'}) 
        self.act_args = kwargs.get('act_args', {'act': 'relu'}) 
        self.conv_args = kwargs.get('conv_args', None)
        self.sampler = kwargs.get('sampler', 'fps')
        self.expansion = kwargs.get('expansion', 4)
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.use_res = kwargs.get('use_res', True)
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)

        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')

        # double width after downsampling.
        channels = []
        for stride in strides:
            if stride != 1:
                width *= 2
            channels.append(width)
        encoder = []
        pe_encoder = nn.ModuleList() #[]
        pe_grouper = []
        for i in range(len(blocks)):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            encoder.append(self._make_enc(
                block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                is_head=i == 0 and strides[i] == 1
            ))
            if i == 0:
                pe_encoder.append(nn.ModuleList())
                pe_grouper.append([])
            else:
                pe_encoder.append(self._make_pe_enc(
                    block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                    is_head=i == 0 and strides[i] == 1
                ))
                pe_grouper.append(create_grouper(group_args))
        self.encoder = nn.Sequential(*encoder)
        self.pe_encoder = pe_encoder #nn.Sequential(*pe_encoder)
        self.pe_grouper = pe_grouper
        self.out_channels = channels[-1]
        self.channel_list = channels

    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_pe_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        ## for PE of this stage
        channels2 = [3, channels]
        convs2 = []
        if blocks > 1:
            for i in range(len(channels2) - 1):  # #layers in each blocks
                convs2.append(create_convblock2d(channels2[i], channels2[i + 1],
                                                norm_args=self.norm_args,
                                                act_args=self.act_args,
                                                **self.conv_args)
                            )
            convs2 = nn.Sequential(*convs2)
            return convs2
        else:
            return nn.ModuleList()

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        group_args.radius = radii[0]
        group_args.nsample = nsample[0]
        layers.append(SetAbstraction(self.in_channels, channels,
                                     self.sa_layers if not is_head else 1, stride,
                                     group_args=group_args,
                                     sampler=self.sampler,
                                     norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args,
                                     is_head=is_head, use_res=self.sa_use_res, **self.aggr_args 
                                     ))
        self.in_channels = channels
        for i in range(1, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res
                                ))
        return nn.Sequential(*layers)

    def forward_cls_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        for i in range(0, len(self.encoder)):
            pe = None
            p0, f0, pe, _ = self.encoder[i]([p0, f0, pe, None])
        return f0.squeeze(-1)

    def forward_seg_feat(self, p0, f0=None, sorted_idx=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        if sorted_idx is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        p, f = [p0], [f0]
        for i in range(0, len(self.encoder)):
            if i == 0:
                pe = None
                _p, _f, _ = self.encoder[i]([p[-1], f[-1], pe, sorted_idx])
            else:
                _p, _f, _ = self.encoder[i][0]([p[-1], f[-1], pe, sorted_idx])
                if self.blocks[i] > 1:
                    # grouping index cache for repeated blocks in the same stage
                    knn_idx = self.pe_grouper[i](_p, _p, None)
                    dp, _ = group_by_idx(_p, _p, None, knn_idx)
                    # conv on neighborhood_dp
                    pe = self.pe_encoder[i](dp)
                    _p, _f, _, _ = self.encoder[i][1:]([_p, _f, pe, knn_idx])
            p.append(_p)
            f.append(_f)
        return p, f

    def forward(self, p0, f0=None, sorted_idx=None):
        return self.forward_seg_feat(p0, f0, sorted_idx)