import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import torch

from vedastd.models import build_model


def main():
    norm_cfg = dict(type='BN')
    cfg = dict(
        type='GModel',
        backbone=dict(
            type='ResNet',
            arch='resnet18',
            replace_stride_with_dilation=None,
            multi_grid=None,
            pretrain=True
        ),
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
                        norm_cfg=None,
                        activation=None,
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
                                      mode='nearest', )
                    ),
                    lateral=dict(
                        from_layer='c4',
                        type='ConvModule',
                        in_channels=256,
                        out_channels=256,
                        kernel_size=1,
                        norm_cfg=None,
                        activation=None,
                    ),
                    post=None,
                    to_layer='p4',
                    fusion_method='add',
                ),
                dict(
                    type='JunctionBlock',
                    top_down=dict(
                        from_layer='p4',
                        upsample=dict(type='Upsample',
                                      scale_factor=2,
                                      mode='nearest', )
                    ),
                    lateral=dict(
                        from_layer='c3',
                        type='ConvModule',
                        in_channels=128,
                        out_channels=256,
                        kernel_size=1,
                        norm_cfg=None,
                        activation=None,
                    ),
                    post=None,
                    to_layer='p3',
                    fusion_method='add',
                ),
                dict(
                    type='JunctionBlock',
                    top_down=dict(
                        from_layer='p3',
                        upsample=dict(type='Upsample',
                                      scale_factor=2,
                                      mode='nearest', )
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
                    post=None,
                    to_layer='p2',
                    fusion_method='add',
                ),
            ],
        ),
        fusion=dict(
            type='FusionBlock',
            method='concat',
            from_layers=['p5', 'p4', 'p3', 'p2'],
            feat_strides=[32, 16, 8, 4],
            in_channels_list=[256, 256, 256, 256],
            out_channels_list=[64, 64, 64, 64],
            upsample=dict(
                type='Upsample',
                scale_factor=2,
                mode='nearest',
            ),
            conv_cfg=dict(type='Conv'),
            norm_cfg=None,
            activation=None,
            inplace=True,
            common_stride=4,
        ),
        head=dict(
            type='DBHead',
            k=10,
            fuse_binary=True,
            adaptive=True,
            thresh=dict(
                name='thresh_map',
                layers=[
                    dict(type='ConvModule', in_channels=257, out_channels=64, kernel_size=3,
                         stride=1, padding=1, bias=False, norm_cfg=norm_cfg),
                    dict(type='Upsample', scale_factor=2, mode='nearest'),
                    dict(type='ConvModule', in_channels=64, out_channels=64, kernel_size=3,
                         stride=1, padding=1, bias=False, norm_cfg=norm_cfg),
                    dict(type='Upsample', scale_factor=2, mode='nearest'),
                    dict(type='ConvModule', in_channels=64, out_channels=64, kernel_size=3,
                         stride=1, padding=1, bias=False, norm_cfg=None, activation=None),
                    dict(type='ConvModule', in_channels=64, out_channels=1, kernel_size=1,
                         stride=1, padding=0, bias=True, norm_cfg=None, activation='sigmoid'),
                ]
            ),
            binary=dict(
                name='binary_map',
                layers=[
                    dict(type='ConvModule', in_channels=256, out_channels=64, kernel_size=3,
                         stride=1, padding=1, bias=False, norm_cfg=norm_cfg),
                    dict(type='Upsample', scale_factor=2, mode='nearest'),
                    dict(type='ConvModule', in_channels=64, out_channels=64, kernel_size=3,
                         stride=1, padding=1, bias=False, norm_cfg=norm_cfg),
                    dict(type='Upsample', scale_factor=2, mode='nearest'),
                    dict(type='ConvModule', in_channels=64, out_channels=64, kernel_size=3,
                         stride=1, padding=1, bias=False, norm_cfg=None, activation=None),
                    dict(type='ConvModule', in_channels=64, out_channels=1, kernel_size=1,
                         stride=1, padding=0, bias=True, norm_cfg=None, activation='sigmoid'),
                ]
            ),
            out_name='thresh_binary_map',
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

    if isinstance(out_gpu, dict):
        for key in out_gpu.keys():
            print(f'diff between cpu and gpu, {key} '
                  f'{torch.max(out_cpu[key] - out_gpu[key].cpu())}')
    else:
        diff = torch.max(out_cpu - out_gpu.cpu())
        print(f'diff between cpu and gpu is {diff}')


if __name__ == '__main__':
    main()
