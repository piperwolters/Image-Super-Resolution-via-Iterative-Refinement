import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import modules
logger = logging.getLogger('base')
####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G(opt):
    output_size = opt['datasets']['output_size'] if 'output_size' in opt['datasets'] else 512

    model_opt = opt['model']
    if model_opt['which_model_G'] == 'ddpm':
        from .ddpm_modules import diffusion, unet
    elif model_opt['which_model_G'] == 'sr3':
        from .sr3_modules import diffusion, unet, unet_3d, aggregate_3d, anchor_aggregate_3d

    if 'unet' in model_opt:
        if ('norm_groups' not in model_opt['unet']) or model_opt['unet']['norm_groups'] is None:
            model_opt['unet']['norm_groups']=32
        model = unet.UNet(
            in_channel=model_opt['unet']['in_channel'],
            out_channel=model_opt['unet']['out_channel'],
            norm_groups=model_opt['unet']['norm_groups'],
            inner_channel=model_opt['unet']['inner_channel'],
            channel_mults=model_opt['unet']['channel_multiplier'],
            attn_res=model_opt['unet']['attn_res'],
            res_blocks=model_opt['unet']['res_blocks'],
            dropout=model_opt['unet']['dropout'],
            image_size=model_opt['diffusion']['image_size']
        )
    elif 'unet_3d' in model_opt:
        model = unet_3d.UNet3D(
            in_channel=model_opt['unet_3d']['in_channel'],
            out_channel=model_opt['unet_3d']['out_channel'],
            channel_mults=model_opt['unet_3d']['channel_multiplier'],
            image_size=model_opt['diffusion']['image_size']
        )
    elif 'aggregate_3d' in model_opt:
        if ('norm_groups' not in model_opt['aggregate_3d']) or model_opt['aggregate_3d']['norm_groups'] is None:
            model_opt['aggregate_3d']['norm_groups']=32
        model = aggregate_3d.UNet(
            in_channel=model_opt['aggregate_3d']['in_channel'],
            out_channel=model_opt['aggregate_3d']['out_channel'],
            norm_groups=model_opt['aggregate_3d']['norm_groups'],
            inner_channel=model_opt['aggregate_3d']['inner_channel'],
            channel_mults=model_opt['aggregate_3d']['channel_multiplier'],
            attn_res=model_opt['aggregate_3d']['attn_res'],
            res_blocks=model_opt['aggregate_3d']['res_blocks'],
            dropout=model_opt['aggregate_3d']['dropout'],
            image_size=model_opt['diffusion']['image_size']
        )
    elif 'anchor_aggregate_3d' in model_opt:
        if ('norm_groups' not in model_opt['anchor_aggregate_3d']) or model_opt['anchor_aggregate_3d']['norm_groups'] is None:
            model_opt['anchor_aggregate_3d']['norm_groups']=32
        model = anchor_aggregate_3d.UNet(
            in_channel=model_opt['anchor_aggregate_3d']['in_channel'],
            out_channel=model_opt['anchor_aggregate_3d']['out_channel'],
            norm_groups=model_opt['anchor_aggregate_3d']['norm_groups'],
            inner_channel=model_opt['anchor_aggregate_3d']['inner_channel'],
            channel_mults=model_opt['anchor_aggregate_3d']['channel_multiplier'],
            attn_res=model_opt['anchor_aggregate_3d']['attn_res'],
            res_blocks=model_opt['anchor_aggregate_3d']['res_blocks'],
            dropout=model_opt['anchor_aggregate_3d']['dropout'],
            image_size=model_opt['diffusion']['image_size']
        )
    else:
        print("Model name not recognized, see model/networks.py to add.")

    netG = diffusion.GaussianDiffusion(
        model,
        image_size=model_opt['diffusion']['image_size'],
        channels=model_opt['diffusion']['channels'],
        loss_type='l1',    # L1 or L2
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train'],
        output_size=output_size,
        use_3d=bool(opt['datasets']['use_3d']),
        is_ddim_sampling=bool(model_opt['diffusion']['is_ddim_sampling']),
        unconditional_guidance_scale=model_opt['diffusion']['unconditional_guidance_scale']
    )

    if opt['phase'] == 'train':
        # init_weights(netG, init_type='kaiming', scale=0.1)
        init_weights(netG, init_type='orthogonal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG
