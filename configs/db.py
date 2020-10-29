# config for inference

norm_cfg = dict(type='BN')

# 9. device
deploy = dict(
    gpu_id='0',
    transforms=[
        dict(type='LongestMaxSize', max_size=640, interpolation='bilinear', p=1),
        dict(type='PadIfNeeded', min_height=640, min_width=640, border_mode='constant',
             value=0),
        dict(type='Normalize', mean=(123.675, 116.280, 103.530),
             std=(58.395, 57.120, 57.375), max_pixel_value=255),
        dict(type='ToTensor'),
    ],
    postprocessor=dict(
        type='Postprocessor',
        min_size=1,
        thresh=0.3,
        box_thresh=0.3,
        name='binary_map',
        unclip_ratio=0.1,
    ),
    model=dict(
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
)
# 2. configuration for train/test
root_workdir = 'workdir/'
dataset_type = 'TxtDataset'
dataset_root = r'D:\DATA_ALL\STD\IC5'

common = dict(
    seed=0,
    logger=dict(
        handlers=(
            dict(type='StreamHandler', level='INFO'),
            dict(type='FileHandler', level='INFO'),
        ),
    ),
    cudnn_deterministic=False,
    cudnn_benchmark=True,
    metric=dict(type='QuadMeasurer', polygon=False),
)

## 2.1 configuration for test
test = dict(
    data=dict(
        dataset=[dict(type=dataset_type,
                      img_root=dataset_root + r'\ch4_training_images',
                      gt_root=dataset_root + r'\ch4_training_localization_transcription_gt',
                      txt_file=dataset_root + r'\train.txt',
                      ignore_tag='###',
                      )],
        transforms=deploy['transforms'],
        dataloader=dict(type='BaseDataloader',
                        batch_size=2,
                        )
    )
)
## 2.2 configuration for train
max_epoch = 200
max_iterations = 200
train = dict(
    data=dict(
        train=dict(
            dataset=[dict(type=dataset_type,
                          img_root=dataset_root + r'\ch4_training_images',
                          gt_root=dataset_root + r'\ch4_training_localization_transcription_gt',
                          txt_file=dataset_root + r'\train.txt',
                          ignore_tag='###',
                          )],
            transforms=[
                dict(type='LongestMaxSize', max_size=640, interpolation='bilinear', p=1),
                dict(type='PadIfNeeded', min_height=640, min_width=640, border_mode='constant',
                     value=0),
                dict(type='KeypointsToPolygon'),
                dict(type='MakeShrinkMap', ratios=[1.0], max_shr=0.6, min_text_size=4, p=1),
                dict(type='MaskMarker', name='gt'),
                dict(type='MakeBorderMap', shrink_ratio=0.4),
                dict(type='MaskMarker', name='border'),
                dict(type='Normalize', mean=(123.675, 116.280, 103.530),
                     std=(58.395, 57.120, 57.375), max_pixel_value=255),
                dict(type='Grouping', channel_first=False),
                dict(type='ToTensor'),
            ],
            collate_fn=dict(type='BaseCollate', stack_keys=['image', 'gt', 'border']),
            dataloader=dict(type='BaseDataloader', batch_size=2),
        ),
        val=dict(
            dataset=[dict(type=dataset_type,
                          img_root=dataset_root + r'\ch4_training_images',
                          gt_root=dataset_root + r'\ch4_training_localization_transcription_gt',
                          txt_file=dataset_root + r'\train.txt',
                          ignore_tag='###',
                          )],
            transforms=deploy['transforms'],
            collate_fn=dict(type='BaseCollate', stack_keys=['image']),
            dataloader=dict(type='BaseDataloader', batch_size=2)
        ),
    ),
    criterion=[
        dict(type='BalanceCrossEntropyLoss', negative_ratio=3.0, eps=1e-6, pred_map='binary_map',
             target='gt', loss_name='bce', loss_weight=5),
        dict(type='MaskL1Loss', pred_map='thresh_map', target='border',
             loss_weight=10, loss_name='mask l1'),
        dict(type='DiceLoss', eps=1e-6, pred_map='thresh_binary_map', target='gt',
             loss_weight=0.1, loss_name='dice loss'),
    ],
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
    ),
    lr_scheduler=dict(
        type='ConstantLR',
    ),
    postprocess=dict(type='Postprocessor',
                     thresh=0.3,
                     box_thresh=0.3,
                     name='binary_map',
                     min_size=1,
                     unclip_ratio=0.1,
                     ),
    max_epochs=max_epoch,
    trainval_ratio=10,
    log_interval=1,
    snapshot_interval=20,
    save_best=True,
    resume=None,
)
