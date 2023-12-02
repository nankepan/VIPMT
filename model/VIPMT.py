import torch
from torch import nn
import torch.nn.functional as F

from model.resnet import *
from model.loss import WeightedDiceLoss
from model.VIPMT_Transformer import VIPMTransformer
from model.backbone_utils import Backbone
from einops import rearrange
import functools


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


class UpsampleDeterministic(nn.Module):
    def __init__(self, upscale=2):
        super(UpsampleDeterministic, self).__init__()
        self.upscale = upscale

    def forward(self, x):
        '''
        x: 4-dim tensor. shape is (batch,channel,h,w)
        output: 4-dim tensor. shape is (batch,channel,self.upscale*h,self.upscale*w)
        '''
        return x[:, :, :, None, :, None] \
            .expand(-1, -1, -1, self.upscale, -1, self.upscale) \
            .reshape(x.size(0), x.size(1), x.size(2) \
                     * self.upscale, x.size(3) * self.upscale)


class Interpolate(nn.Module):

    def __init__(self, channel: int, scale_factor: int):
        super().__init__()
        # assert 'mode' not in kwargs and 'align_corners' not in kwargs and 'size' not in kwargs
        assert isinstance(scale_factor, int) and scale_factor > 1 and scale_factor % 2 == 0
        self.scale_factor = scale_factor
        kernel_size = scale_factor + 1  # keep kernel size being odd
        self.weight = nn.Parameter(
            torch.empty((1, 1, kernel_size, kernel_size), dtype=torch.float32).expand(channel, -1, -1, -1)
        )
        self.conv = functools.partial(
            F.conv2d, weight=self.weight, bias=None, padding=scale_factor // 2, groups=channel
        )
        with torch.no_grad():
            self.weight.fill_(1 / (kernel_size * kernel_size))

    def forward(self, t):
        if t is None:
            return t
        return self.conv(F.interpolate(t, scale_factor=self.scale_factor, mode='nearest'))

    @staticmethod
    def naive(t: torch.Tensor, size: [int, int], **kwargs):
        if t is None or t.shape[2:] == size:
            return t
        else:
            assert 'mode' not in kwargs and 'align_corners' not in kwargs
            return F.interpolate(t, size, mode='nearest', **kwargs)


def overalliou(pred, target):
    """
    param: pred of size [N x H x W]
    param: target of size [N x H x W]
    """
    assert len(pred.shape) == 3 and pred.shape == target.shape
    inter = torch.min(pred, target).sum(2).sum(1).sum(0)
    union = torch.max(pred, target).sum(2).sum(1).sum(0)
    iou = inter / union
    return iou


def meaniou(pred, target):
    """
    param: pred of size [N x H x W]
    param: target of size [N x H x W]
    """
    assert len(pred.shape) == 3 and pred.shape == target.shape
    N = pred.size(0)
    inter = torch.min(pred, target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)
    iou = torch.sum(inter / union) / N
    return iou


def mask_iou_loss(pred, mask, loss, mode='mean'):
    N, F, H, W = mask.shape
    for i in range(N):
        if mode == 'mean':
            loss += (1.0 - meaniou(pred[i], mask[i]))
        elif mode == 'overall':
            loss += (1.0 - overalliou(pred[i], mask[i]))
        else:
            raise RuntimeError("mode should be mean or overall]")
    loss = loss / N
    return loss


def maskiou(mask1, mask2):
    b = mask1.size(0)
    mask1 = mask1.view(b, -1)
    mask2 = mask2.view(b, -1)
    area1 = mask1.sum(dim=1, keepdim=True)
    area2 = mask2.sum(dim=1, keepdim=True)
    inter = ((mask1 + mask2) == 2).sum(dim=1, keepdim=True)
    union = (area1 + area2 - inter)
    for a in range(b):
        if union[a][0] == torch.tensor(0):
            union[a][0] = torch.tensor(1)
    maskiou = inter / union
    return maskiou


def get_iou_label(pred, mask):
    N, F, H, W = mask.shape
    iou_gt = []
    for i in range(F):
        iou = maskiou(pred[:, i, :, :], mask[:, i, :, :])
        iou_gt.append(iou)
    iou_gt = torch.cat(iou_gt, dim=1)
    return iou_gt


class Score(nn.Module):
    def __init__(self, input_chan):
        super(Score, self).__init__()
        input_channels = input_chan
        self.conv1 = nn.Conv2d(input_channels, 256, 3, 2, 1)
        self.conv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 2, 1)

        self.fc1 = nn.Linear(256 * 8 * 14, 1024)  # fc layers
        self.fc2 = nn.Linear(1024, 1)

        self.donw_samp = nn.Conv2d(256, 256, 3, 2, 1)
        self.fuse_conv1 = nn.Conv2d(3840, 256, 3, 1, 1)
        self.fuse_conv2 = nn.Conv2d(256, 256, 3, 1, 1)

        # self.gav = nn.AdaptiveAvgPool2d(1)        #global average pooling layers
        # self.fc = nn.Linear(256, 1)

        for i in [self.conv1, self.conv2, self.conv3, self.conv4, self.donw_samp, self.fuse_conv1, self.fuse_conv2]:
            nn.init.kaiming_normal_(i.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(i.bias, 0)

        for i in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(i.weight, a=1)
            nn.init.constant_(i.bias, 0)

    def forward(self, f1, f2, f3, f4, y):  # 1/4 256, 1/8 512, 1/8 1024, 1/8 2048, 1/1 1
        f1 = F.relu(self.donw_samp(f1))
        x = torch.cat((f1, f2, f3, f4), dim=1)
        x = F.relu(self.fuse_conv1(x))
        x = F.relu(self.fuse_conv2(x))
        y = rearrange(y, 'b t c h w -> (b t) c h w')
        x = torch.cat((x, y), dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        # x = self.gav(x)               #the method of gav
        # x = x.view(x.size(0), -1)
        # x = F.leaky_relu(self.fc(x))
        return x


class VIPMT(nn.Module):
    def __init__(self, args, layers=50, classes=2, shot=5, reduce_dim=384,
                 criterion=WeightedDiceLoss(), with_transformer=True, trans_multi_lvl=1):
        super(VIPMT, self).__init__()
        assert layers in [50, 101]
        assert classes > 1
        self.layers = layers
        self.criterion = criterion
        self.mask_iou_loss = mask_iou_loss
        self.compute_iou = get_iou_label
        self.shot = shot
        self.with_transformer = with_transformer
        if self.with_transformer:
            self.trans_multi_lvl = trans_multi_lvl
        self.reduce_dim = reduce_dim

        self.print_params()

        in_fea_dim = 1024 + 512

        drop_out = 0.5

        self.adjust_feature_supp = nn.Sequential(
            nn.Conv2d(in_fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_out),
        )
        self.adjust_feature_qry = nn.Sequential(
            nn.Conv2d(in_fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_out),
        )

        self.high_avg_pool = nn.AdaptiveAvgPool1d(reduce_dim)

        prior_channel = 1
        self.qry_merge_feat = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 + prior_channel, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        if self.with_transformer:
            self.supp_merge_feat = nn.Sequential(
                nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )

            self.transformer = VIPMTransformer(embed_dims=reduce_dim, shot=self.shot)
            self.merge_multi_lvl_reduce = nn.Sequential(
                nn.Conv2d(reduce_dim * 1, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            )
            self.merge_multi_lvl_sum = nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )

        self.merge_res = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.ini_cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )

        self.init_weights()

        self.score_head = Score(261)  # 256 + 5
        self.score_loss = nn.L1Loss(reduction='sum')

        self.backbone = Backbone(args, 'resnet{}'.format(layers), train_backbone=False, return_interm_layers=True,
                                 dilation=[False, True, True])

        self.aux_loss = nn.BCEWithLogitsLoss()
        self.aux_loss2 = nn.BCEWithLogitsLoss(reduction='none')

        self.upsample1 = Interpolate(channel=2, scale_factor=8)
        self.upsample2 = Interpolate(channel=2, scale_factor=8)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def print_params(self):
        repr_str = self.__class__.__name__
        repr_str += f'(backbone layers={self.layers}, '
        repr_str += f'reduce_dim={self.reduce_dim}, '
        repr_str += f'shot={self.shot}, '
        repr_str += f'with_transformer={self.with_transformer})'
        print(repr_str)
        return repr_str

    def forward(self, x, y, s_x, s_y, clip=1):
        batch_size, q_t, c, h, w = x.size()
        _, s_t, _, _, _, = s_y.size()  # b t 1 h w
        t = s_t, q_t
        s_y = s_y.squeeze(2)
        y = y.squeeze(2)
        img_size = x.size()[-2:]
        y = y.view(-1, *img_size)

        # backbone feature extraction
        qry_bcb_fts = self.backbone(x.view(-1, 3, *img_size))  # 1/4 256, 1/8 512, 1/8 1024, 1/8 2048
        if clip == 1:
            supp_bcb_fts = self.backbone(s_x.view(-1, 3, *img_size))
        else:
            for k, v in s_x.items():
                s_x[k] = rearrange(v, 'b t c h w -> (b t) c h w')
            supp_bcb_fts = s_x

        query_feat23 = torch.cat([qry_bcb_fts['1'], qry_bcb_fts['2']], dim=1)
        supp_feat23 = torch.cat([supp_bcb_fts['1'], supp_bcb_fts['2']], dim=1)
        query_feat23 = self.adjust_feature_qry(query_feat23)
        supp_feat23 = self.adjust_feature_supp(supp_feat23)

        fts_size = query_feat23.shape[-2:]
        supp_mask = F.interpolate((s_y == 1).view(-1, *img_size).float().unsqueeze(1), size=(fts_size[0], fts_size[1]),
                                  mode='nearest')

        # global feature extraction
        supp_feat_list = []
        r_supp_feat = supp_feat23.view(batch_size, s_t, -1, fts_size[0], fts_size[1])
        for st in range(s_t):
            mask = (s_y[:, st, :, :] == 1).float().unsqueeze(1)
            mask = F.interpolate(mask, size=(fts_size[0], fts_size[1]), mode='nearest')
            tmp_supp_feat = r_supp_feat[:, st, ...]
            tmp_supp_feat = Weighted_GAP(tmp_supp_feat, mask)  # [b, c, 1, 1]
            supp_feat_list.append(tmp_supp_feat)
        global_supp_pp = supp_feat_list[0]
        if s_t > 1:
            for i in range(1, len(supp_feat_list)):
                global_supp_pp += supp_feat_list[i]
            global_supp_pp /= len(supp_feat_list)
            multi_supp_pp = Weighted_GAP(supp_feat23, supp_mask)  # [bs*s_t, c, 1, 1]
        else:
            multi_supp_pp = global_supp_pp

        # prior generation
        query_feat_high = qry_bcb_fts['3'].view(batch_size, -1, 2048, fts_size[0], fts_size[1])
        supp_feat_high = supp_bcb_fts['3'].view(batch_size, -1, 2048, fts_size[0], fts_size[1])
        corr_query_mask = self.generate_prior(query_feat_high, supp_feat_high, s_y, fts_size)

        # feature mixing
        global_supp_pp = global_supp_pp.unsqueeze(1).repeat(1, q_t, 1, 1, 1).expand(-1, -1, -1, fts_size[0],
                                                                                    fts_size[1])
        query_cat_feat = [query_feat23,
                          global_supp_pp.view(-1, query_feat23.shape[1], fts_size[0], fts_size[1]),
                          corr_query_mask.view(-1, 1, fts_size[0], fts_size[1])]
        query_feat = self.qry_merge_feat(torch.cat(query_cat_feat, dim=1))  # [b*t, c, h, w]

        query_feat_out = self.merge_res(query_feat) + query_feat
        init_out = self.ini_cls(query_feat_out)  # [b*t, 2, h, w]
        init_mask = init_out.max(1)[1]  # [b*t, h, w]
        init_mask = init_mask.view(batch_size, q_t, fts_size[0], fts_size[1])

        to_merge_fts = [supp_feat23, multi_supp_pp.expand(-1, -1, fts_size[0], fts_size[1])]
        aug_supp_feat = torch.cat(to_merge_fts, dim=1)
        aug_supp_feat = self.supp_merge_feat(aug_supp_feat)  # [b*t, c, h, w]

        fused_query_feat, qry_outputs_mask_list_fg, sup_outputs_mask_list, multividmask = \
            self.transformer(query_feat, y.float(),
                             aug_supp_feat, s_y.clone().float(), init_mask.detach(), t,
                             rearrange(query_feat23, '(b t) c h w -> b t c h w', b=batch_size))

        fused_query_feat = self.merge_multi_lvl_reduce(fused_query_feat)
        fused_query_feat = self.merge_multi_lvl_sum(fused_query_feat) + fused_query_feat

        # Output Part
        out = self.cls(fused_query_feat)  # [b*t 2 h w]

        query_mask = out.view(batch_size, q_t, 2, out.size(-2), out.size(-1)).max(2)[1]
        PFE_mask = self.generate_PFEmask(query_feat_high, supp_feat_high,
                                         s_y, query_mask)
        score = self.score_head(qry_bcb_fts['0'], qry_bcb_fts['1'], qry_bcb_fts['2'], qry_bcb_fts['3'],
                                PFE_mask.detach())

        out = self.upsample1(out)

        score = rearrange(score, '(b t) c -> b t c', b=batch_size, t=q_t)
        score = score.squeeze(2)

        if self.training:
            # calculate loss
            main_dice_loss1 = self.criterion(out, y.long())
            main_meaniou_loss1 = torch.zeros_like(main_dice_loss1)
            main_overalliou_loss1 = torch.zeros_like(main_dice_loss1)
            main_meaniou_loss1 = self.mask_iou_loss(
                torch.nn.Softmax(dim=2)(out.view(batch_size, q_t, 2, *img_size))[:, :, 1, :, :],
                y.view(batch_size, q_t, *img_size).long(),
                main_meaniou_loss1, mode='mean')
            main_overalliou_loss1 = self.mask_iou_loss(
                torch.nn.Softmax(dim=2)(out.view(batch_size, q_t, 2, *img_size))[:, :, 1, :, :],
                y.view(batch_size, q_t, *img_size).long(),
                main_overalliou_loss1, mode='overall')
            main_loss = main_dice_loss1 + main_meaniou_loss1 + main_overalliou_loss1

            mask_for_pred = out.view(batch_size, q_t, 2, *img_size).max(2)[1]
            iou_gt = self.compute_iou(mask_for_pred.detach(),
                                      y.view(batch_size, q_t, *img_size).long())
            loss_score = self.score_loss(score, iou_gt) / batch_size

            init_out = self.upsample2(init_out)
            main_dice_loss2 = self.criterion(init_out, y.long())
            main_meaniou_loss2 = torch.zeros_like(main_dice_loss2)
            main_overalliou_loss2 = torch.zeros_like(main_dice_loss2)
            main_meaniou_loss2 = self.mask_iou_loss(
                torch.nn.Softmax(dim=2)(init_out.view(batch_size, q_t, 2, *img_size))[:, :, 1, :, :],
                y.view(batch_size, q_t, *img_size).long(),
                main_meaniou_loss2, mode='mean')
            main_overalliou_loss2 = self.mask_iou_loss(
                torch.nn.Softmax(dim=2)(init_out.view(batch_size, q_t, 2, *img_size))[:, :, 1, :, :],
                y.view(batch_size, q_t, *img_size).long(),
                main_overalliou_loss2, mode='overall')
            main_loss2 = main_dice_loss2 + main_meaniou_loss2 + main_overalliou_loss2

            aux_loss_q = torch.zeros_like(main_loss)
            aux_loss_s = torch.zeros_like(main_loss)
            multivid_contrast_loss = torch.zeros_like(main_loss)
            for qy_id, qry_out in enumerate(qry_outputs_mask_list_fg):
                q_gt = F.interpolate((y.view(batch_size, q_t, *img_size) == 1) * 1.0, size=qry_out.size()[-2:],
                                     mode='nearest')
                aux_loss_q = aux_loss_q + self.aux_loss(qry_out, q_gt)
            aux_loss_q = aux_loss_q / len(qry_outputs_mask_list_fg)
            for st_id, supp_out in enumerate(sup_outputs_mask_list):
                s_gt = F.interpolate((s_y == 1) * 1.0, size=supp_out.size()[-2:], mode='nearest')
                aux_loss_s = aux_loss_s + self.aux_loss(supp_out, s_gt)
            aux_loss_s = aux_loss_s / len(sup_outputs_mask_list)

            for i_ in range(multividmask.shape[1]):
                qry_out = multividmask[:, i_, :, :, :]
                q_gt = F.interpolate((y.view(batch_size, q_t, *img_size) == 1) * 1.0, size=qry_out.size()[-2:],
                                     mode='nearest')
                contrast_loss_map = self.aux_loss2(qry_out, 1-q_gt)
                contrast_loss_map = contrast_loss_map * q_gt
                contrast_loss_map = rearrange(contrast_loss_map, 'b t h w -> (b t h w)')
                denominator_map = rearrange(q_gt, 'b t h w -> (b t h w)')
                contrast_loss = contrast_loss_map.sum() / (denominator_map.sum() + 1e-7)
                multivid_contrast_loss = multivid_contrast_loss + contrast_loss
            multivid_contrast_loss = multivid_contrast_loss / (multividmask.shape[1] + 1e-7)  # 1e-7 for bs < 2

            erfa = 0.3
            aux_loss = erfa * aux_loss_q + (1 - erfa) * aux_loss_s

            for k, v in qry_bcb_fts.items():
                qry_bcb_fts[k] = rearrange(v, '(b t) c h w -> b t c h w', b=batch_size).detach()
            if clip == 1:
                for k, v in supp_bcb_fts.items():
                    supp_bcb_fts[k] = rearrange(v, '(b t) c h w -> b t c h w', b=batch_size).detach()
            else:
                supp_bcb_fts = None

            return out.view(batch_size, q_t, 2, *img_size).max(2)[1], 0.7 * main_loss + 0.3 * main_loss2, aux_loss, multivid_contrast_loss, loss_score, qry_bcb_fts, supp_bcb_fts
        else:
            return out.view(batch_size, q_t, 2, *img_size).max(2)[1], score

    def generate_prior(self, mul_query_feat_high, supp_feat_high, s_y, fts_size):
        bsize, q_t, q_c, q_h, q_w = mul_query_feat_high.size()[:]
        s_t = supp_feat_high.shape[1]
        corr_query_mask_list = []
        mul_corr_query_mask_list = []
        cosine_eps = 1e-7
        for qt in range(q_t):
            query_feat_high = mul_query_feat_high[:, qt, :, :, :]
            for st in range(s_t):
                tmp_mask = (s_y[:, st, :, :] == 1).float().unsqueeze(1)
                tmp_mask = F.interpolate(tmp_mask, size=(fts_size[0], fts_size[1]), mode='nearest')

                tmp_supp_feat = supp_feat_high[:, st, ...] * tmp_mask
                q = self.high_avg_pool(query_feat_high.flatten(2).transpose(-2, -1))  # [bs, h*w, c]
                s = self.high_avg_pool(tmp_supp_feat.flatten(2).transpose(-2, -1))  # [bs, h*w, c]

                tmp_query = q
                tmp_query = tmp_query.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
                tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

                tmp_supp = s
                tmp_supp = tmp_supp.contiguous()
                tmp_supp = tmp_supp.contiguous()
                tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

                similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
                similarity = similarity.max(1)[0].view(bsize, q_h * q_w)
                similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                            similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
                corr_query = similarity.view(bsize, 1, q_h, q_w)
                corr_query_mask_list.append(corr_query)
            corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)
            mul_corr_query_mask_list.append(corr_query_mask.unsqueeze(1))
        mul_corr_query_mask = torch.cat(mul_corr_query_mask_list, 1)
        return mul_corr_query_mask

    def generate_PFEmask(self, mul_query_feat_high, supp_feat_high, s_y, query_mask):
        bsize, q_t, q_c, q_h, q_w = mul_query_feat_high.size()[:]
        s_t = supp_feat_high.shape[1]
        s_h, s_w = supp_feat_high.shape[-2:]
        mul_corr_query_mask_list = []
        cosine_eps = 1e-7
        for qt in range(q_t):
            corr_query_mask_list = []
            query_feat_high = mul_query_feat_high[:, qt, :, :, :]
            q_mask = (query_mask[:, qt, :, :] == 1).float().unsqueeze(1)
            q_mask = F.interpolate(q_mask, size=(q_h, q_w), mode='nearest')
            corr_query_mask_list.append(q_mask)

            query_feat_high_fg = query_feat_high * q_mask
            q = self.high_avg_pool(query_feat_high_fg.flatten(2).transpose(-2, -1))  # [bs, h*w, c]
            tmp_query = q
            tmp_query = tmp_query.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            query_feat_high_bg = query_feat_high * (1 - q_mask)
            q_bg = self.high_avg_pool(query_feat_high_bg.flatten(2).transpose(-2, -1))  # [bs, h*w, c]
            tmp_query_bg = q_bg
            tmp_query_bg = tmp_query_bg.contiguous().permute(0, 2, 1)  # [bs, c, h*w]
            tmp_query_norm_bg = torch.norm(tmp_query_bg, 2, 1, True)

            tmp_mask = F.interpolate((s_y == 1).float(), size=(s_h, s_w), mode='nearest')
            tmp_supp_feat = supp_feat_high * tmp_mask.unsqueeze(2)
            tmp_supp_feat = rearrange(tmp_supp_feat, 'b t c h w -> b c t h w')
            s = self.high_avg_pool(tmp_supp_feat.flatten(2).transpose(-2, -1))  # [bs, t*h*w, c]
            tmp_supp = s
            tmp_supp = tmp_supp.contiguous()
            tmp_supp = tmp_supp.contiguous()
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity_qbg2s = torch.bmm(tmp_supp, tmp_query_bg) / (torch.bmm(tmp_supp_norm, tmp_query_norm_bg) + cosine_eps)
            similarity_qbg2s = similarity_qbg2s.max(1)[0].view(bsize, q_h * q_w)
            similarity_qbg2s = (similarity_qbg2s - similarity_qbg2s.min(1)[0].unsqueeze(1)) / (
                    similarity_qbg2s.max(1)[0].unsqueeze(1) - similarity_qbg2s.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query_qbg2s = similarity_qbg2s.view(bsize, 1, q_h, q_w)
            corr_query_mask_list.append(corr_query_qbg2s)

            similarity_qfg2s = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            similarity_qfg2s = similarity_qfg2s.max(1)[0].view(bsize, q_h * q_w)
            similarity_qfg2s = (similarity_qfg2s - similarity_qfg2s.min(1)[0].unsqueeze(1)) / (
                    similarity_qfg2s.max(1)[0].unsqueeze(1) - similarity_qfg2s.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query_qfg2s = similarity_qfg2s.view(bsize, 1, q_h, q_w)
            corr_query_mask_list.append(corr_query_qfg2s)

            similarity_qfg_qbg = torch.bmm(tmp_query_bg.transpose(-2, -1), tmp_query) / (torch.bmm(tmp_query_norm_bg.transpose(-2, -1), tmp_query_norm) + cosine_eps)
            similarity_qfg2qbg = similarity_qfg_qbg.max(1)[0].view(bsize, q_h * q_w)
            similarity_qfg2qbg = (similarity_qfg2qbg - similarity_qfg2qbg.min(1)[0].unsqueeze(1)) / (
                    similarity_qfg2qbg.max(1)[0].unsqueeze(1) - similarity_qfg2qbg.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query_qfg2qbg = similarity_qfg2qbg.view(bsize, 1, q_h, q_w)
            corr_query_mask_list.append(corr_query_qfg2qbg)

            similarity_qbg2qfg = similarity_qfg_qbg.max(2)[0].view(bsize, q_h * q_w)
            similarity_qbg2qfg = (similarity_qbg2qfg - similarity_qbg2qfg.min(1)[0].unsqueeze(1)) / (
                    similarity_qbg2qfg.max(1)[0].unsqueeze(1) - similarity_qbg2qfg.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query_qbg2qfg = similarity_qbg2qfg.view(bsize, 1, q_h, q_w)
            corr_query_mask_list.append(corr_query_qbg2qfg)

            corr_query_mask = torch.cat(corr_query_mask_list, dim=1)
            mul_corr_query_mask_list.append(corr_query_mask.unsqueeze(1))
        mul_corr_query_mask = torch.cat(mul_corr_query_mask_list, dim=1)  # b t 5 h w
        return mul_corr_query_mask
