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
batch_size = 2

# transforms
train_transforms = [
    dict(type='MakeShrinkMap', ratios=[0.4], max_shr=20, min_text_size=8, prefix='seg'),
    dict(type='MakeShrinkMap', ratios=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4], max_shr=20, min_text_size=8,
         prefix='kernel'),
    dict(type='MakeBorderMap', shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7),
    dict(type='Resize', keep_ratio=True, size=(448, 448), img_mode='nearest', mask_mode='nearest'),
    # dict(type='RandomRotation', angles=(-10, 10), p=0.5),
    dict(type='Canvas', size=(448, 448), img_v=0, mask_v=0),
    # dict(type='RandomFlip', p=0.5, horizontal=True, vertical=True),
    dict(type='FilterKeys',
         need_keys=['seg_mask', 'seg_map', 'polygon', 'boarder_mask', 'boarder_map', 'shape', 'input', 'tags',
                    'ratio']),
    dict(type='ToTensor', keys=['seg_mask', 'seg_map', 'boarder_mask', 'boarder_map', 'input']),
    # dict(type='Normalize', key='input'),
]

test_transforms = [
    dict(type='MakeShrinkMap', ratios=[0.4], max_shr=20, min_text_size=8, prefix='seg'),
    dict(type='MakeBorderMap', shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7),
    dict(type='Resize', keep_ratio=True, size=(448, 448), img_mode='nearest', mask_mode='nearest'),
    dict(type='Canvas', size=(448, 448), img_v=0, mask_v=0),
    dict(type='FilterKeys',
         need_keys=['seg_mask', 'seg_map', 'polygon', 'boarder_mask', 'boarder_map', 'shape', 'input', 'tags',
                    'ratio']),
    dict(type='ToTensor', keys=['seg_mask', 'seg_map', 'boarder_mask', 'boarder_map', 'input']),
    # dict(type='Normalize', key='input'),
]

dataset = [dict(type='TxtDataset',
                img_root=r'D:\DB-master\dataset\ours\train_images',
                gt_root=r'D:\DB-master\dataset\ours\train_gts',
                txt_file=r'D:\DB-master\dataset\ours\train.txt',
                )]

collect_fns = dict(
    val=dict(type='BaseCollate'),
    test=dict(type='BaseCollate'),
)

train_dataloader = dict(type='BaseDataloader', batch_size=batch_size)
test_dataloader = dict(type='BaseDataloader', batch_size=1)

data = dict(
    train=dict(
        transforms=train_transforms,
        datasets=dataset,
        loader=train_dataloader,
    ),
    val=dict(
        transforms=test_transforms,
        datasets=dataset,
        loader=test_dataloader,
    ),
)

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

criterion = [
    dict(type='BalanceCrossEntropyLoss', negative_ratio=3.0, eps=1e-6, pred_map='binary_map', gt_map='seg_map',
         gt_mask='seg_mask', loss_name='bce', loss_weight=5),
    dict(type='MaskL1Loss', pred_map='thresh_map', gt_map='boarder_map', gt_mask='boarder_mask',
         loss_weight=10, loss_name='mask l1'),
    dict(type='DiceLoss', eps=1e-6, pred_map='thresh_binary_map', gt_map='seg_map', gt_mask='seg_mask',
         loss_weight=0.1, loss_name='dice loss'),
]

postprocessor = dict(
    type='Postprocessor',
    debug=False,
    resize=True,
    thresh=0.3,
    box_thresh=0.7,
    max_candidates=100,
)
# postprocess = dict(
#     train=dict(
#
#     ),
#     val=dict(
#
#     ),
#     test=dict(
#
#     ),
# )
optimizer = dict(type='Adam', lr=0.001)

max_epoch = 1200
lr_scheduler = dict(type='PolyLR', max_epochs=max_epoch)
resume = None
# 8. runner_old
max_iterations = 300000
runner = dict(
    type='Runner',
    epochs=max_epoch,
    iterations=max_iterations,
    trainval_ratio=10,
    snapshot_interval=20000,
    grad_clip=0,
)

# 9. device
gpu_id = '0'
