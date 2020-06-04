# work dir
root_workdir = 'workdir/'

# seed
seed = 1111

# 1. logging
logger = dict(
    handlers=(
        dict(type='StreamHandler', level='INFO'),
        dict(type='FileHandler', level='INFO'),
    ),
)

# 2. data
batch_size = 192

# transforms
transforms = [
    dict(type='MakeShrinkMap', ratios=[0.4], max_shr=20, min_text_size=8, prefix='seg'),
    dict(type='MakeShrinkMap', ratios=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4], max_shr=20, min_text_size=8,
         prefix='kernel'),
    dict(type='MakeBoarderMap', shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7),
    dict(type='Resize', keep_ratio=True, size=(600, 400), img_mode='nearest', mask_mode='nearest'),
    dict(type='RandomRotation', angles=(-10, 10), p=0.5),
    dict(type='RandomFlip', p=0.5, horizontal=True, vertical=True),
    dict(type='FilterKeys'),
    dict(type='ToTensor'),
    dict(type='Normalize', key='input'),
]

dataset = [dict(type='TxtDataset',
                img_root=r'D:\DB-master\dataset\ours\train_images',
                gt_root=r'D:\DB-master\dataset\ours\train_gts',
                txt_file=r'D:\DB-master\dataset\ours\train_list.txt',
                )]

norm_cfg = dict(type='BN')
model = dict(
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

            ]
        ),
        binary=dict(
            name='binary_map',
            layers=[
                dict(),
            ]
        ),
        out_name='thresh_binary_map',
    )
)
