import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib import Grapher, act_layer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch.autograd import Variable

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x

class KLA(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.scale_ = 14 ** 0.5  # N= 14*14
        embed_lens = 512
        self.k_w = nn.Linear(dim, embed_lens, bias=qkv_bias)  # 768 -> 512
        self.v_w = nn.Linear(dim, embed_lens, bias=qkv_bias)  # 768 -> 512
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_lens, dim)  # 512 -> 768
        self.proj_drop = nn.Dropout(proj_drop)
        self.topk = 9

    def forward(self, x, q):
        B, N, C = x.shape
        class_num, e_d = q.shape
        q = q.repeat(B, 1, 1)  # B class_num 512
        k = self.k_w(x)  # B N C: B 196 512
        v = self.v_w(x)  # B N C: B 196 512
        attn = (q @ k.transpose(-2, -1)) * self.scale_  # B class_num 196
        ####################################### topK
        topk, indices = torch.topk(attn, self.topk)
        attn_ = Variable(torch.zeros_like(attn), requires_grad=False)
        attn_ = attn_.type(attn.dtype)
        attn_ = attn_.to(attn.device)
        attn_ = attn_.scatter(-1, indices, topk)
        #######################################
        attn_ = attn_ / attn_.norm(dim=-1, keepdim=True) * attn_.shape[-1]
        attn_ = attn_.softmax(dim=-1)
        attn_topk = self.attn_drop(attn_)

        x = (attn_topk @ v)  #B class_num 512
        x = self.proj(x)  # B class_num 768
        x = self.proj_drop(x)
        return x

class WB(torch.nn.Module):
    def __init__(self, opt):
        super(WB, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2 * k, self.n_blocks)]  # number of knn's k
        print('num_knn', num_knn)
        max_dilation = 196 // max(num_knn)
        gcn_p_length = opt.gcn_len
        if opt.use_dilation:
            self.LIN = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                              bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                      ) for i in range(self.n_blocks)])
        else:
            self.LIN = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                              bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                      ) for i in range(self.n_blocks)])
        # 32
        self.PG_1 = Seq(nn.Conv2d(channels, 32, 1, bias=True),  # 1024
                             nn.BatchNorm2d(32),
                             act_layer(act),
                             nn.Conv2d(32, 12 * gcn_p_length * 768, 1, bias=True),
                             nn.BatchNorm2d(12 * gcn_p_length * 768),
                             nn.Dropout(0.1)
                             )

        self.PG_2 = Seq(nn.Conv2d(channels, 32, 1, bias=True),  # 1024
                             nn.BatchNorm2d(32),
                             act_layer(act),
                             nn.Conv2d(32, 768, 1, bias=True)
                             )
        self.KLA = KLA(dim=768, qkv_bias=False, attn_drop=0., proj_drop=0.)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs, q):
        x = inputs
        for i in range(self.n_blocks):
            x = self.LIN[i](x)  # b 768 14 14

        x_pooling = F.adaptive_avg_pool2d(x, 1)  # b 768 1 1
        B, C, p1, p2 = x.shape  # B 768 14 14
        x = x.reshape(B, C, -1)
        x = x.permute(0, 2, 1)  # B 196 768

        x_att = self.KLA(x, q)  # B, class_num, 768

        x1 = self.PG_2(x_pooling)  # b 768 1 1
        x2 = self.PG_1(x1)  # b 768*12*2 1 1
        x2 = x2.squeeze(-1).squeeze(-1)
        return x1.squeeze(-1).squeeze(-1), x2, x_att

    def inference(self, inputs):

        x = inputs
        for i in range(self.n_blocks):
            x = self.LIN[i](x)  # b 768 14 14
        # print('x:', x.shape)
        x_pooling = F.adaptive_avg_pool2d(x, 1)  # b 768 1 1
        ######################
        B, C, p1, p2 = x.shape  # B 768 14 14
        x = x.reshape(B, C, -1)
        x = x.permute(0, 2, 1)  # B 196 768

        x1 = self.PG_2(x_pooling)  # b 768 1 1
        x2 = self.PG_1(x1)  # b 768*12*2 1 1
        # print('xx:', x.shape)
        x2 = x2.squeeze(-1).squeeze(-1)
        return x1.squeeze(-1).squeeze(-1), x2




def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 768, 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'patch16_224': _cfg(
        crop_pct=0.9,
    ),
}

def White_box_module(pretrained=False, use_stochastic=False, gcn_len=2, **kwargs):
    class OptInit:
        def __init__(self, num_classes=768, drop_path_rate=0.0, drop_rate=0.0, num_knn=9, **kwargs):
            self.k = num_knn  # neighbor num (default:9)
            self.conv = 'mr'  # graph conv layer {edge, mr}
            self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}
            self.norm = 'batch'  # batch or instance normalization {batch, instance}
            self.bias = True  # bias of conv layer True or False
            self.n_blocks = 1  # number of basic blocks in the backbone
            self.n_filters = 768  # number of channels of deep features
            self.n_classes = num_classes  # Dimension of out_channels
            self.dropout = drop_rate  # dropout rate
            self.use_dilation = True  # use dilated knn or not
            self.epsilon = 0.2  # stochastic epsilon for gcn
            self.use_stochastic = use_stochastic # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.gcn_len = gcn_len

    opt = OptInit(**kwargs)
    model = WB(opt)
    model.default_cfg = default_cfgs['patch16_224']
    return model




