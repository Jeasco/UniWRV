import torch.nn as nn
import torch
import blocks as blocks
from quantize import GumbelQuantize
from submodules import DeformableAttnBlock, DeformableAttnBlock_FUSION
# from positional_encodings import PositionalEncodingPermute3D
from torch.nn.init import xavier_uniform_, constant_


def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 4

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]

## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
    def forward(self, x):
        return self.body(x)

class WPGM(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super(WPGM, self).__init__()

        self.mapping_network = nn.Sequential(
            nn.functional.adaptive_avg_pool2d(a, (1,1)),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(inplace=True)
        )

        self.weather_piror_bank = GumbelQuantize(dim, dim,
                                       n_embed=20,
                                       kl_weight=1.0e-08, temp_init=1.0,
                                       remap=None)

        self.feature_extractor = [blocks.ResBlock(dim, dim, kernel_size=kernel_size, stride=1)
                               for _ in range(3)]


    def forward(self, x):
        piror, emb_loss, info = self.weather_piror_bank(self.mapping_network(x))
        x = x + piror
        x = self.feature_extractor(x)
        return x


class WPGM_output_piror(nn.Module):
    def __init__(self, dim, kernel_size = 3):
        super(WPGM_output_piror, self).__init__()

        self.mapping_network = nn.Sequential(
            nn.functional.adaptive_avg_pool2d(a, (1,1)),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(inplace=True)
        )

        self.weather_piror_bank = GumbelQuantize(dim, dim,
                                       n_embed=20,
                                       kl_weight=1.0e-08, temp_init=1.0,
                                       remap=None)

        self.feature_extractor = [blocks.ResBlock(dim, dim, kernel_size=kernel_size, stride=1)
                               for _ in range(3)]


    def forward(self, x):
        piror, emb_loss, info = self.weather_piror_bank(self.mapping_network(x))
        x = x + piror
        x = self.feature_extractor(x)
        return x, piror

class UniWRV(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, n_resblock=3, n_feat=32,
                 kernel_size=3, feat_in=False, n_in_feat=32,num_blocks=[2, 4, 4],):
        super(UniWRV, self).__init__()
        print("Creating Video Restoration Net")

        self.feat_in = feat_in

        InBlock = []
        if not feat_in:
            InBlock.extend([nn.Sequential(
                nn.Conv2d(in_channels, n_feat, kernel_size=3, stride=1,
                          padding=3 // 2),
                nn.LeakyReLU(0.1,inplace=True)
            )])
            print("The input of SRN is image")
        else:
            InBlock.extend([nn.Sequential(
                nn.Conv2d(n_in_feat, n_feat, kernel_size=3, stride=1, padding=3 // 2),
                nn.LeakyReLU(0.1,inplace=True)
            )])
            print("The input of SRN is feature")
        InBlock.extend([blocks.ResBlock(n_feat, n_feat, kernel_size=3, stride=1)
                        for _ in range(3)])

        self.encoder_1 = nn.Sequential(*[WPGM(dim=dim) for _ in range(num_blocks[0])-1])
        self.encoder_3.append(WPGM_output_piror(dim=dim ))
        self.down_1 = Downsample(dim)
        self.encoder_2 = nn.Sequential(*[WPGM(dim=dim * 2) for _ in range(num_blocks[1])-1])
        self.encoder_3.append(WPGM_output_piror(dim=dim * 2))
        self.down_2 = Downsample(dim * 2)
        self.encoder_3 = nn.Sequential(*[WPGM(dim=dim * 4) for _ in range(num_blocks[2])-1])
        self.encoder_3.append(WPGM_output_piror(dim=dim * 4))

        self.reduce_3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1, bias=False)
        self.decoder_3 = nn.Sequential(*[WPGM(dim=dim * 4) for _ in range(num_blocks[2])])
        self.up_2 = Upsample(dim * 4)
        self.reduce_2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1, bias=False)
        self.decoder_2 = nn.Sequential(*[WPGM(dim=dim * 2) for _ in range(num_blocks[1])])
        self.up_1 = Upsample(dim * 2)
        self.reduce_1 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.decoder_1 = nn.Sequential(*[WPGM(dim=dim) for _ in range(num_blocks[0])])

        OutBlock = [blocks.ResBlock(n_feat, n_feat, kernel_size=kernel_size, stride=1)
                    for _ in range(n_resblock)]
        OutBlock.append(
            nn.Conv2d(n_feat, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        )

        self.inBlock_t = nn.Sequential(*InBlock)

        self.outBlock = nn.Sequential(*OutBlock)

        self.DRA_1 = DeformableAttnBlock_FUSION(n_heads=4, d_model=256,n_levels=3,n_points=12)
        self.DRA_2 = DeformableAttnBlock_FUSION(n_heads=4, d_model=256, n_levels=3, n_points=12)
        self.DRA_3 = DeformableAttnBlock_FUSION(n_heads=4, d_model=256, n_levels=3, n_points=12)
        self.DRA_4 = DeformableAttnBlock_FUSION(n_heads=4, d_model=256, n_levels=3, n_points=12)
        
        # self.pos_em  = PositionalEncodingPermute3D(3)
        self.motion_branch = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=2*n_feat * 4, out_channels=96//2, kernel_size=3, stride=1, padding=8, dilation=8),
                    nn.LeakyReLU(0.1,inplace=True),
                    torch.nn.Conv2d(in_channels=96//2, out_channels=64//2, kernel_size=3, stride=1, padding=16, dilation=16),
                    nn.LeakyReLU(0.1,inplace=True),
                    torch.nn.Conv2d(in_channels=64//2, out_channels=32//2, kernel_size=3, stride=1, padding=1, dilation=1),
                    nn.LeakyReLU(0.1,inplace=True),
        )
        self.motion_out = torch.nn.Conv2d(in_channels=32//2, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
        constant_(self.motion_out.weight.data, 0.)
        constant_(self.motion_out.bias.data, 0.)
    def compute_flow(self, frames):
        n, t, c, h, w = frames.size()
        frames_1 = frames[:, :-1, :, :, :].reshape(-1, c, h, w)
        frames_2 = frames[:, 1:, :, :, :].reshape(-1, c, h, w)
        flows_forward = self.estimate_flow(frames_1, frames_2).view(n, t-1, 2, h, w)
        # print(flows_forward.shape)
        flows_backward = self.estimate_flow(frames_2,frames_1).view(n, t-1, 2, h, w)

        return flows_forward,flows_backward
    def estimate_flow(self,frames_1, frames_2):
        return self.motion_out(self.motion_branch(torch.cat([frames_1, frames_2],1)))
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B, -1, H, W)
        x, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(x)
        x = x.view(B, T, C, H, W)
        b, n, c, h, w = x.size()

        inblock = self.inBlock_t(x.view(b*n,c,h,w))

        out_encoder_1, piror_1 = self.encoder_1(iinblock)

        input_encoder_2 = self.down_1(out_encoder_1)
        out_encoder_2, piror_2 = self.encoder_2(input_encoder_2)

        input_encoder_3 = self.down_2(out_encoder_2)
        out_encoder_3, piror_3 = self.encoder_3(input_encoder_3)

        out_encoder_3 = out_encoder_3.view(b,n,128,h//4,w//4)
        
        flow_forward,flow_backward = self.compute_flow(out_encoder_3)
        
        frame, srcframe = out_encoder_3, out_encoder_3

        prior = piror_1.repeat((4,1)) + piror_2.repeat((2,1)) + piror_3

        dra_layer1 = self.DRA_1(frame[:,1],srcframe,flow_forward,flow_backward,prior)
        dra_layer2 = self.DRA_2(dra_layer1, srcframe, flow_forward, flow_backward,prior)
        dra_layer3 = self.DRA_3(dra_layer2, srcframe, flow_forward, flow_backward,prior)
        dra_layer4 = self.DRA_4(dra_layer3, srcframe, flow_forward, flow_backward,prior)

        out_decoder_3 = self.decoder_3(dra_layer4)

        input_decoder_2 = self.up_2(out_decoder_3)
        input_decoder_2 = torch.cat([input_decoder_2, out_encoder_2], 1)
        input_decoder_2 = self.reduce_2(input_decoder_2)
        out_decoder_2 = self.decoder_2(input_decoder_2)

        input_decoder_1 = self.up_1(out_decoder_2)
        input_decoder_1 = torch.cat([input_decoder_1, out_encoder_1], 1)
        input_decoder_1 = self.reduce_1(input_decoder_1)
        out_decoder_1 = self.decoder_1(input_decoder_1)

        first_scale_outBlock = self.outBlock(out_decoder_1+out_encoder_1.view(b,n,32,h,w)[:,1])

        first_scale_outBlock = pad_tensor_back(first_scale_outBlock, pad_left, pad_right, pad_top, pad_bottom)

        return first_scale_outBlock, flow_forward, flow_backward

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

if __name__ == '__main__':
    import os

    torch.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    from ptflops import get_model_complexity_info


    model = UniWRV().cuda()


    macs, params = get_model_complexity_info(model, (9, 256, 256), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
