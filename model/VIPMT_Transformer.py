from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import math
from model.positional_encoding import SinePositionalEncoding
from einops import rearrange


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int): The number of fully-connected layers in FFNs.
        dropout (float): Probability of an element to be zeroed. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 dropout=0.0,
                 add_residual=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
                             f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.dropout = dropout
        self.activate = nn.ReLU(inplace=True)

        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels, bias=False), self.activate,
                    nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims, bias=False))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + self.dropout(out)


class QSCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert num_heads == 1, "currently only implement num_heads==1"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_fc = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self.drop_prob = 0.1

    def forward(self, prototype, q_x, qry_attn_mask):
        q = self.q_fc(prototype)
        k = self.k_fc(q_x)
        v = self.v_fc(q_x)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if qry_attn_mask is not None:
            qry_attn_mask = (~qry_attn_mask).unsqueeze(-2).float()
            qry_attn_mask = qry_attn_mask * -10000.0
            attn = attn + qry_attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SupMemCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert num_heads==1, "currently only implement num_heads==1"
        self.num_heads  = num_heads
        head_dim        = dim // num_heads
        self.scale      = qk_scale or head_dim ** -0.5

        self.q_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_fc = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_fc = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop  = nn.Dropout(attn_drop)
        self.proj       = nn.Linear(dim, dim, bias=False)
        self.proj_drop  = nn.Dropout(proj_drop)

        self.drop_prob = 0.1

    def forward(self, prototype, s_x, sup_attn_mask, m_x=None, mem_attn_mask=None):
        q = self.q_fc(prototype)
        s_k = self.k_fc(s_x)
        s_v = self.v_fc(s_x)
        sup_attn = (q @ s_k.transpose(-2, -1)) * self.scale
        sup_attn_mask = (~sup_attn_mask).unsqueeze(-2).float()
        sup_attn_mask = sup_attn_mask * -10000.0
        sup_attn = sup_attn + sup_attn_mask

        if m_x is not None:
            m_k = self.k_fc(m_x)
            m_v = self.v_fc(m_x)
            mem_attn = (q @ m_k.transpose(-2, -1)) * self.scale
            mem_attn_mask = (~mem_attn_mask).unsqueeze(-2).float()
            mem_attn_mask = mem_attn_mask * -10000.0
            mem_attn = mem_attn + mem_attn_mask

            dim = sup_attn.shape[-1]

            attn = torch.cat((sup_attn, mem_attn), dim=-1)
            attn = attn.softmax(dim=-1)
            sup_attn = attn[:, :, :dim]
            mem_attn = attn[:, :, dim:]

            mem_attn = self.attn_drop(mem_attn)
            m_x = (mem_attn @ m_v)
            m_x = self.proj(m_x)
            m_x = self.proj_drop(m_x)
        else:
            sup_attn = sup_attn.softmax(dim=-1)

        sup_attn = self.attn_drop(sup_attn)
        s_x = (sup_attn @ s_v)
        s_x = self.proj(s_x)
        s_x = self.proj_drop(s_x)

        return s_x, m_x


class MySelfAttention(nn.Module):  # PVT
    def __init__(self, dim, heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., sr_ratio=4):
        super(MySelfAttention, self).__init__()

        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class VIPMTransformer(nn.Module):
    def __init__(self,
                 embed_dims=384,
                 num_heads=1,
                 su_num_layers=3,
                 num_layers=5,
                 num_levels=1,
                 use_ffn=True,
                 dropout=0.1,
                 shot=5,
                 ):
        super(VIPMTransformer, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.su_num_layers = su_num_layers
        self.num_layers = num_layers
        self.num_levels = num_levels
        self.use_ffn = use_ffn
        self.feedforward_channels = embed_dims * 3
        self.dropout = dropout
        self.shot = shot
        self.use_cross = False
        self.use_self = True

        if self.use_cross:
            self.sigmoid_cross_layers = []

        self.query_decoder_cross_attention_layers = []
        self.support_memory_decoder_cross_attention_layers = []
        self.query_decoder_cross_attention_layers_frame = []
        self.query_layer_norms = []
        self.support_memory_layer_norms = []
        self.query_layer_norms_frame = []
        self.query_ffns = []
        self.support_memory_ffns = []
        self.query_ffns_frame = []
        self.query_layer_norms1 = []
        self.support_memory_layer_norms1 = []
        self.query_layer_norms1_frame = []
        self.update_q = []
        self.res_q = []
        self.qry_self_layers = []
        self.layer_norms_q1 = []
        self.ffns_q = []
        self.layer_norms_q2 = []

        for c_id in range(self.num_layers):
            self.query_decoder_cross_attention_layers.append(
                QSCrossAttention(embed_dims, attn_drop=self.dropout, proj_drop=self.dropout),
            )
            self.support_memory_decoder_cross_attention_layers.append(
                SupMemCrossAttention(embed_dims, attn_drop=self.dropout, proj_drop=self.dropout),
            )
            self.query_decoder_cross_attention_layers_frame.append(
                QSCrossAttention(embed_dims, attn_drop=self.dropout, proj_drop=self.dropout),
            )

            self.query_layer_norms.append(nn.LayerNorm(embed_dims))
            self.support_memory_layer_norms.append(nn.LayerNorm(embed_dims))
            self.query_layer_norms_frame.append(nn.LayerNorm(embed_dims))

            self.query_ffns.append(FFN(embed_dims, self.feedforward_channels, dropout=self.dropout))
            self.support_memory_ffns.append(FFN(embed_dims, self.feedforward_channels, dropout=self.dropout))
            self.query_ffns_frame.append(FFN(embed_dims, self.feedforward_channels, dropout=self.dropout))

            self.query_layer_norms1.append(nn.LayerNorm(embed_dims))
            self.support_memory_layer_norms1.append(nn.LayerNorm(embed_dims))
            self.query_layer_norms1_frame.append(nn.LayerNorm(embed_dims))

            self.update_q.append(nn.Sequential(
                nn.Conv2d(embed_dims * 2, embed_dims, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True)
            ))
            self.res_q.append(nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))

            self.qry_self_layers.append(
                MySelfAttention(dim=embed_dims, attn_drop=self.dropout, proj_drop=self.dropout)
            )
            self.layer_norms_q1.append(nn.LayerNorm(embed_dims))
            self.ffns_q.append(FFN(embed_dims, self.feedforward_channels, dropout=self.dropout))
            self.layer_norms_q2.append(nn.LayerNorm(embed_dims))

        self.query_decoder_cross_attention_layers = nn.ModuleList(self.query_decoder_cross_attention_layers)
        self.support_memory_decoder_cross_attention_layers = nn.ModuleList(self.support_memory_decoder_cross_attention_layers)
        self.query_decoder_cross_attention_layers_frame = nn.ModuleList(self.query_decoder_cross_attention_layers_frame)
        self.query_layer_norms = nn.ModuleList(self.query_layer_norms)
        self.support_memory_layer_norms = nn.ModuleList(self.support_memory_layer_norms)
        self.query_layer_norms_frame = nn.ModuleList(self.query_layer_norms_frame)
        self.query_ffns = nn.ModuleList(self.query_ffns)
        self.support_memory_ffns = nn.ModuleList(self.support_memory_ffns)
        self.query_ffns_frame = nn.ModuleList(self.query_ffns_frame)
        self.query_layer_norms1 = nn.ModuleList(self.query_layer_norms1)
        self.support_memory_layer_norms1 = nn.ModuleList(self.support_memory_layer_norms1)
        self.query_layer_norms1_frame = nn.ModuleList(self.query_layer_norms1_frame)
        self.update_q = nn.ModuleList(self.update_q)
        self.res_q = nn.ModuleList(self.res_q)
        self.qry_self_layers = nn.ModuleList(self.qry_self_layers)
        self.layer_norms_q1 = nn.ModuleList(self.layer_norms_q1)
        self.ffns_q = nn.ModuleList(self.ffns_q)
        self.layer_norms_q2 = nn.ModuleList(self.layer_norms_q2)

        self.positional_encoding = SinePositionalEncoding(embed_dims // 2, normalize=True)
        self.level_embed = nn.Parameter(torch.rand(num_levels, embed_dims))
        nn.init.xavier_uniform_(self.level_embed)

        self.proj_drop = nn.Dropout(dropout)

        self.mask_embed_clip = FFN(embed_dims, embed_dims, dropout=self.dropout)
        self.decoder_norm_clip = nn.LayerNorm(embed_dims)
        self.mask_embed_frame = FFN(embed_dims, embed_dims, dropout=self.dropout)
        self.decoder_norm_frame = nn.LayerNorm(embed_dims)
        self.prototype = nn.Embedding(1, embed_dims)

    def init_weights(self, distribution='uniform'):
        """Initialize the transformer weights."""
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_reference_points(self, spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points.unsqueeze(2).repeat(1, 1, len(spatial_shapes), 1)
        return reference_points

    def get_qry_flatten_input(self, x, qry_masks):
        src_flatten = []
        pos_embed_flatten = []
        for lvl in range(self.num_levels):
            src = x[lvl]
            bs, c, h, w = src.shape  # here bs is bs*t

            src = src.flatten(2).permute(0, 2, 1)  # [bs, c, h*w] -> [bs, h*w, c]
            src_flatten.append(src)

            if qry_masks is not None:
                qry_mask = qry_masks[lvl]
                qry_valid_mask = []
                qry_mask = F.interpolate(
                    qry_mask.unsqueeze(1), size=(h, w), mode='nearest').squeeze(1)
                for img_id in range(bs):
                    qry_valid_mask.append(qry_mask[img_id] == 255)
                qry_valid_mask = torch.stack(qry_valid_mask, dim=0)
            else:
                qry_valid_mask = torch.zeros((bs, h, w))

            pos_embed = self.positional_encoding(qry_valid_mask)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            pos_embed_flatten.append(pos_embed)

        src_flatten = torch.cat(src_flatten, 1)  # [bs, num_elem, c]
        pos_embed_flatten = torch.cat(pos_embed_flatten, dim=1)  # [bs, num_elem, c]

        return src_flatten, None, pos_embed_flatten, None, None

    def get_supp_flatten_input(self, s_x, supp_mask, s_t):
        s_x_flatten = []
        supp_obj_mask = []
        supp_mask = F.interpolate(supp_mask, size=s_x.shape[-2:], mode='nearest')  # [bs*shot, h, w]
        supp_mask = supp_mask.view(-1, s_t, s_x.size(2), s_x.size(3))
        s_x = s_x.view(-1, s_t, s_x.size(1), s_x.size(2), s_x.size(3))

        for st_id in range(s_x.size(1)):
            supp_obj_mask_s = []
            for img_id in range(s_x.size(0)):
                obj_mask = supp_mask[img_id, st_id, ...] == 1
                if obj_mask.sum() == 0:  # To avoid NaN
                    obj_mask[obj_mask.size(0) // 2 - 1:obj_mask.size(0) // 2 + 1,
                    obj_mask.size(1) // 2 - 1:obj_mask.size(1) // 2 + 1] = True
                if (obj_mask == False).sum() == 0:  # To avoid NaN
                    obj_mask[0, 0] = False
                    obj_mask[-1, -1] = False
                    obj_mask[0, -1] = False
                    obj_mask[-1, 0] = False
                supp_obj_mask_s.append(obj_mask)

            supp_obj_mask_s = torch.stack(supp_obj_mask_s, dim=0)
            supp_obj_mask_s = (supp_obj_mask_s == 1).flatten(1)  # [bs, n]
            supp_obj_mask.append(supp_obj_mask_s.unsqueeze(1))

            s_x_s = s_x[:, st_id, ...]
            s_x_s = s_x_s.flatten(2).permute(0, 2, 1)  # [bs, c, h*w] -> [bs, h*w, c]
            s_x_flatten.append(s_x_s.unsqueeze(1))

        s_x_flatten = torch.cat(s_x_flatten, 1)  # [bs, t, h*w, c]
        supp_mask_flatten = torch.cat(supp_obj_mask, 1)

        return s_x_flatten, None, supp_mask_flatten

    def forward(self, x, qry_masks, s_x, supp_mask, init_mask, t, query_feat23):
        if not isinstance(x, list):
            x = [x]
        if not isinstance(qry_masks, list):
            qry_masks = [qry_masks.clone() for _ in range(self.num_levels)]

        assert len(x) == len(qry_masks) == self.num_levels
        bs_t, c, h, w = x[0].size()
        bs = supp_mask.size()[0]
        s_t, q_t = t

        x_flatten, qry_valid_masks_flatten, pos_embed_flatten, spatial_shapes, level_start_index = self.get_qry_flatten_input(x, qry_masks)

        s_x, supp_valid_mask, supp_mask_flatten = self.get_supp_flatten_input(s_x, supp_mask.clone(), s_t)

        qry_outputs_mask_list_fg = []
        sup_outputs_mask_list = []

        prototype = self.prototype.weight.unsqueeze(0).repeat(bs, 1, 1)  # b 1 c
        q = x_flatten.view(bs, q_t, h, w, c)  # b t h w c
        s_x = s_x.view(bs, s_t, h, w, c)  # b t h w c
        pos = pos_embed_flatten  # (b t) (h w) c
        qry_attn_mask = init_mask  # b t h w
        supp_mask_flatten = supp_mask_flatten.view(bs, s_t, h, w)  # b t h w

        q = q.flatten(1, 3)  # b (t h w) c
        qry_attn_mask = qry_attn_mask.flatten(1)  # b (t h w)

        if s_t == 5:
            s = s_x.flatten(1, 3)  # b (t h w) c
            sup_attn_mask = supp_mask_flatten.flatten(1)  # b (t h w)
            m = None
            mem_attn_mask = None
        else:
            s = s_x[:, :5].flatten(1, 3)
            sup_attn_mask = supp_mask_flatten[:, :5].flatten(1)  # b (t h w)
            m = s_x[:, 5:].flatten(1, 3)
            mem_attn_mask = supp_mask_flatten[:, 5:].flatten(1)  # b (t h w)

        g2f_mask_bs_iter = []
        for c_id in range(self.num_layers):
            sup_prototype, mem_prototype = self.support_memory_decoder_cross_attention_layers[c_id](prototype, s, sup_attn_mask, m, mem_attn_mask)
            sup_prototype = self.support_memory_layer_norms[c_id](sup_prototype + prototype)
            sup_prototype = self.support_memory_ffns[c_id](sup_prototype)
            sup_prototype = self.support_memory_layer_norms1[c_id](sup_prototype)  # b 1 c
            if mem_prototype is not None:
                mem_prototype = self.support_memory_layer_norms[c_id](mem_prototype + prototype)
                mem_prototype = self.support_memory_ffns[c_id](mem_prototype)
                mem_prototype = self.support_memory_layer_norms1[c_id](mem_prototype)  # b 1 c

            qry_prototype = self.query_decoder_cross_attention_layers[c_id](prototype, q, qry_attn_mask) + prototype
            qry_prototype = self.query_layer_norms[c_id](qry_prototype)
            qry_prototype = self.query_ffns[c_id](qry_prototype)
            qry_prototype = self.query_layer_norms1[c_id](qry_prototype)  # b 1 c

            prototype_cliplevel = qry_prototype + sup_prototype + prototype
            if mem_prototype is not None:
                prototype_cliplevel = prototype_cliplevel + mem_prototype

            qry_outputs_fg, qry_attn_mask, sup_outputs_mask, _, prototype_cliplevel = \
                self.forward_prediction_heads_cliplevel(prototype_cliplevel, q, s_x.flatten(1, 3))

            g2f_mask_bs = []
            for b_ in range(bs):
                qf23_b = rearrange(query_feat23[b_, :, :, :, :], 't c h w -> (t h w) c')
                prototype_clip_b = torch.cat((prototype_cliplevel[:b_, :, :], prototype_cliplevel[b_+1:, :, :]),  dim=0).squeeze(1)  # b-1 c
                g2f_mask = torch.einsum("nc,bc->bn", qf23_b, prototype_clip_b).unsqueeze(0)  # 1 b-1 (t h w)
                g2f_mask_bs.append(g2f_mask)
            g2f_mask_bs = torch.cat(g2f_mask_bs, dim=0)  # b b-1 (t h w)
            g2f_mask_bs_iter.append(g2f_mask_bs)

            qry_outputs_fg = rearrange(qry_outputs_fg, 'b (t h w) -> b t h w', t=q_t, h=h, w=w)
            qry_outputs_mask_list_fg.append(qry_outputs_fg)
            sup_outputs_mask_list.append(rearrange(sup_outputs_mask, 'b (t h w) -> b t h w', t=s_t, h=h, w=w))
            qry_attn_mask = rearrange(qry_attn_mask, 'b (t h w) -> (b t) (h w)', t=q_t, h=h, w=w)

            prototype_cliptoframe = prototype_cliplevel.unsqueeze(1).repeat(1, q_t, 1, 1).flatten(0, 1)  # bt 1 c
            q = rearrange(q, 'b (t h w) c -> (b t) (h w) c', h=h, w=w)  # bt n c
            prototype_framelevel = self.query_decoder_cross_attention_layers_frame[c_id](prototype_cliptoframe, q, qry_attn_mask) + prototype_cliptoframe
            prototype_framelevel = self.query_layer_norms_frame[c_id](prototype_framelevel)
            prototype_framelevel = self.query_ffns_frame[c_id](prototype_framelevel)
            prototype_framelevel = self.query_layer_norms1_frame[c_id](prototype_framelevel)  # bt 1 c
            prototype_framelevel = prototype_framelevel + prototype_cliptoframe

            q = rearrange(q, '(b t) hw c -> b t hw c', b=bs)  # b t n c
            prototype_framelevel = rearrange(prototype_framelevel, '(b t) n c -> b t n c', b=bs)  # b t 1 c
            qry_outputs_mask, qry_attn_mask, prototype_framelevel = \
                self.forward_prediction_heads_framelevel(prototype_framelevel, q)
            qry_outputs_mask_list_fg.append(rearrange(qry_outputs_mask, 'b t (h w) -> b t h w', h=h, w=w))
            qry_attn_mask = rearrange(qry_attn_mask, 'b t (h w) -> b (t h w)', h=h, w=w)

            tmp_prototype = prototype_framelevel.expand_as(q)

            tmp_q = rearrange(torch.cat((q, tmp_prototype), dim=-1), 'b t (h w) c -> (b t) c h w', h=h, w=w)
            q = self.update_q[c_id](tmp_q)
            q = q + self.res_q[c_id](q)

            q = rearrange(q, 'bt c h w -> bt (h w) c')
            q = q + self.proj_drop(
                self.qry_self_layers[c_id](q+pos, H=h, W=w)
            )
            q = self.layer_norms_q1[c_id](q)
            q = self.ffns_q[c_id](q)
            q = self.layer_norms_q2[c_id](q)
            q = q.view(bs, q_t, h, w, c).flatten(1, 3)

            prototype = torch.mean(prototype_framelevel, dim=1)

        qry_feat = rearrange(q, 'b (t h w) c -> (b t) c h w', h=h, w=w)
        multividmask = torch.cat(g2f_mask_bs_iter, dim=1)  # b (b-1)*5 (t h w)
        multividmask = rearrange(multividmask, 'b d (t h w) -> b d t h w', t=q_t, h=h, w=w)

        return qry_feat, qry_outputs_mask_list_fg, sup_outputs_mask_list, multividmask

    def forward_prediction_heads_cliplevel(self, output, query_mask_features, support_mask_features):
        decoder_output = self.decoder_norm_clip(output)
        mask_embed = self.mask_embed_clip(decoder_output)

        query_outputs_mask = torch.einsum("bqc,bnc->bqn", mask_embed, query_mask_features).squeeze(-2)  # b (t h w)
        query_attn_mask = (query_outputs_mask.sigmoid() >= 0.5).bool()
        query_attn_mask = query_attn_mask.detach()

        support_outputs_mask = torch.einsum("bqc,bnc->bqn", mask_embed, support_mask_features).squeeze(-2)  # b (t h w)
        support_attn_mask = (support_outputs_mask.sigmoid() >= 0.5).bool()
        support_attn_mask = support_attn_mask.detach()

        return query_outputs_mask, query_attn_mask, support_outputs_mask, support_attn_mask, mask_embed

    def forward_prediction_heads_framelevel(self, output, query_mask_features):
        decoder_output = self.decoder_norm_frame(output)
        mask_embed = self.mask_embed_frame(decoder_output)

        query_outputs_mask = torch.einsum("btqc,btnc->btqn", mask_embed, query_mask_features).squeeze(-2)  # b t (h w)
        query_attn_mask = (query_outputs_mask.sigmoid() >= 0.5).bool()
        query_attn_mask = query_attn_mask.detach()

        return query_outputs_mask, query_attn_mask, mask_embed




