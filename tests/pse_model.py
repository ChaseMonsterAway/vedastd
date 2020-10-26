import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import torch

from vedastd.models import build_model


def main():
    norm_cfg = dict(type='BN')
    kernel_num = 6
    cfg = dict(
        type='GModel',
        backbone=dict(
            type='ResNet',
            arch='resnet18',
            replace_stride_with_dilation=None,
            multi_grid=None,
            pretrain=True
        ),
        # resnet18: 64,128,256,512
        # resnet50: 256,512,1024,2048
        enhance=dict(
            type='GFPN',
            neck=[
                dict(
                    type='JunctionBlock',
                    top_down=None,
                    lateral=dict(
                        from_layer='c5',
                        type='ConvModule',
                        in_channels=512,
                        out_channels=256,
                        kernel_size=1,
                        norm_cfg=norm_cfg,
                        activation='relu',
                    ),
                    post=None,
                    to_layer='p5',
                ),
                dict(
                    type='JunctionBlock',
                    top_down=dict(
                        from_layer='p5',
                        upsample=dict(type='Upsample',
                                      scale_factor=2,
                                      mode='bilinear',
                                      align_corners=True,
                                      )
                    ),
                    lateral=dict(
                        from_layer='c4',
                        type='ConvModule',
                        in_channels=256,
                        out_channels=256,
                        kernel_size=1,
                        norm_cfg=norm_cfg,
                        activation='relu',
                    ),
                    post=dict(
                        type='ConvModule',
                        in_channels=256,
                        out_channels=256,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=norm_cfg,
                        activation='relu',
                    ),
                    to_layer='p4',
                    fusion_method='add',
                ),
                dict(
                    type='JunctionBlock',
                    top_down=dict(
                        from_layer='p4',
                        upsample=dict(type='Upsample',
                                      scale_factor=2,
                                      mode='bilinear',
                                      align_corners=True,
                                      )
                    ),
                    lateral=dict(
                        from_layer='c3',
                        type='ConvModule',
                        in_channels=128,
                        out_channels=256,
                        kernel_size=1,
                        norm_cfg=norm_cfg,
                        activation='relu',
                    ),
                    post=dict(
                        type='ConvModule',
                        in_channels=256,
                        out_channels=256,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=norm_cfg,
                        activation='relu',
                    ),
                    to_layer='p3',
                    fusion_method='add',
                ),
                dict(
                    type='JunctionBlock',
                    top_down=dict(
                        from_layer='p3',
                        upsample=dict(type='Upsample',
                                      scale_factor=2,
                                      mode='bilinear',
                                      align_corners=True, )
                    ),
                    lateral=dict(
                        from_layer='c2',
                        type='ConvModule',
                        in_channels=64,
                        out_channels=256,
                        kernel_size=1,
                        norm_cfg=None,
                        activation=None,
                    ),
                    post=dict(
                        type='ConvModule',
                        in_channels=256,
                        out_channels=256,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=norm_cfg,
                        activation='relu',
                    ),
                    to_layer='p2',
                    fusion_method='add',
                ),
            ],
        ),
        fusion=dict(
            type='UpsampleFusion',
            method='concat',
            from_layers=['p5', 'p4', 'p3', 'p2'],
            feat_strides=[32, 16, 8, 4],
            upsample=dict(
                type='Upsample',
                mode='bilinear',
                align_corners=True,
            ),
            common_stride=4,
        ),
        head=dict(
            type='PseHead',
            scale=1,
            name=('pred_text_map', 'pred_kernels_map'),
            layers=[
                dict(type='ConvModule',
                     in_channels=1024,
                     out_channels=256,
                     kernel_size=3,
                     stride=1,
                     padding=1,
                     norm_cfg=norm_cfg,
                     activation='relu',
                     ),
                dict(type='Upsample',
                     scale_factor=4,
                     mode='bilinear',
                     align_corners=True,
                     ),
                dict(type='ConvModule',
                     in_channels=256,
                     out_channels=kernel_num,
                     kernel_size=1,
                     stride=1,
                     padding=0,
                     norm_cfg=None,
                     activation=None,
                     ),
            ],
        )
    )
    model = build_model(cfg)
    print(model)

    # dummy input
    dummy_input = torch.rand(size=(1, 3, 640, 640), dtype=torch.float32)
    # on cpu
    out_cpu = model(dummy_input)
    # on gpu
    out_gpu = model.cuda()(dummy_input.cuda())


if __name__ == '__main__':
    main()
