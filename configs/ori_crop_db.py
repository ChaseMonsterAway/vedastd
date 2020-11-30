# config for inference

norm_cfg = dict(type='BN')

# 9. device
deploy = dict(
    gpu_id='7',
    transforms=[
        dict(type='LongestMaxSize', max_size=1280, interpolation='bilinear', p=1),
        dict(type='PadorResize', min_height=1280, min_width=1280, border_mode='constant',
             value=0),
        dict(type='Normalize', mean=(0.485, 0.456, 0.406),
             std=(0.229, 0.224, 0.225), max_pixel_value=255),
        dict(type='ToTensor'),
    ],
    postprocessor=dict(
        type='Postprocessor',
        min_size=4,
        thresh=0.3,
        box_thresh=0.5,
        name='binary_map',
        unclip_ratio=1.5,
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
                        bias=False,
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
                        bias=False,
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
                        bias=False,
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
                        bias=False,
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
            multi_conv=False,
            upsample=dict(
                type='Upsample',
                scale_factor=2,
                mode='nearest',
            ),
            conv_cfg=dict(type='Conv'),  # bias=False
            norm_cfg=None,
            activation=None,
            inplace=True,
            bias=False,
            common_stride=4,
        ),
        head=dict(
            type='DBHead',
            k=50,
            fuse_binary=False,
            adaptive=True,
            thresh=dict(
                name='thresh_map',
                layers=[
                    dict(type='ConvModule', in_channels=256, out_channels=64, kernel_size=3,
                         stride=1, padding=1, bias=False, norm_cfg=norm_cfg),
                    dict(type='ConvModule', conv_cfg=dict(type='ConvTranspose'),
                         in_channels=64, out_channels=64, kernel_size=2,
                         stride=2, norm_cfg=norm_cfg),
                    dict(type='ConvModule', conv_cfg=dict(type='ConvTranspose'),
                         in_channels=64, out_channels=1, kernel_size=2,
                         stride=2, norm_cfg=None, activation='sigmoid'),
                ],
            ),
            binary=dict(
                name='binary_map',
                layers=[
                    dict(type='ConvModule', in_channels=256, out_channels=64, kernel_size=3,
                         stride=1, padding=1, bias=False, norm_cfg=norm_cfg),
                    dict(type='ConvModule', conv_cfg=dict(type='ConvTranspose'),
                         in_channels=64, out_channels=64, kernel_size=2,
                         stride=2, norm_cfg=norm_cfg),
                    dict(type='ConvModule', conv_cfg=dict(type='ConvTranspose'),
                         in_channels=64, out_channels=1, kernel_size=2,
                         stride=2, norm_cfg=None, activation='sigmoid'),
                ],
            ),
            out_name='thresh_binary_map',
        )
    ),
)
# 2. configuration for train/test
root_workdir = 'workdir/'
dataset_type = 'TxtDataset'
dataset_root = r'/DATA/home/junsun/code/scene_text_detection/vedastd/data'

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
test_postprocessors=[dict(
        type='Postprocessor',
        min_size=3,
        thresh=thresh / 100.0,
        box_thresh=box_thresh / 100.0,
        name='binary_map',
        unclip_ratio=1.5,
        ) 
        for thresh in list(range(30, 70, 5))
        for box_thresh in list(range(thresh,75, 5))
        # for unclip_ratio in list(range(13, 17, 1))
        # for min_size in list(range(3,4,1))
  ]
test = dict(
    data=dict(
        dataset=[dict(type=dataset_type,
                      img_root=dataset_root,
                      gt_root=dataset_root,
                      txt_file=dataset_root + '/ic15_test.txt',
                      ignore_tag='1',
                      )],
        transforms=deploy['transforms'],
        collate_fn=dict(type='BaseCollate', stack_keys=['image']),
        dataloader=dict(type='BaseDataloader',
                        batch_size=1,
                        num_workers=4,
                        )
    ),
    postprocessor=dict(type='SearchPostprocessor',
                       post_processors=test_postprocessors,
                      )
)
## 2.2 configuration for train
batch_size = 16
max_epoch = 1200
# max_iterations = 1200
train = dict(
    data=dict(
        train=dict(
            dataset=[dict(type=dataset_type,
                          img_root=dataset_root,
                          gt_root=dataset_root,
                          txt_file=dataset_root + r'/ic15_train.txt',
                          ignore_tag='1',
                          )],
            transforms=[
                dict(type='RandomScale', scale_range=(0.5, 3.0), interpolation='bilinear', p=1),
                dict(type='IAAFliplr', p=0.5),
                dict(type='Rotate', limit=10, border_mode='constant', value=0),
                # dict(type='LongestMaxSize', max_size=640, interpolation='bilinear', p=1),
                # dict(type='PadIfNeeded', min_height=256, min_width=256, border_mode='constant', value=0),
                dict(type='RandomCropBasedOnBox', p=1.0),
                dict(type='PadorResize', min_height=640, min_width=640, border_mode='constant', value=0),
                dict(type='KeypointsToPolygon'),
                dict(type='MakeShrinkMap', ratios=[0.4], max_shr=20, min_text_size=8, p=1),
                dict(type='MaskMarker', name='gt'),
                dict(type='MakeBorderMap', shrink_ratio=0.4),
                dict(type='MaskMarker', name='border'),
                dict(type='Normalize', mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225), max_pixel_value=255),
                dict(type='Grouping', channel_first=False),
                dict(type='ToTensor'),
            ],
            collate_fn=dict(type='BaseCollate', stack_keys=['image', 'gt', 'border']),
            dataloader=dict(type='BaseDataloader', batch_size=batch_size, num_workers=4, shuffle=True),
        ),
        val=dict(
            dataset=[dict(type=dataset_type,
                          img_root=dataset_root,
                          gt_root=dataset_root,
                          txt_file=dataset_root + r'/ic15_test.txt',
                          ignore_tag='1',
                          )],
            transforms=deploy['transforms'],
            collate_fn=dict(type='BaseCollate', stack_keys=['image']),
            dataloader=dict(type='BaseDataloader', batch_size=batch_size, num_workers=1),
        ),
    ),
    criterion=[
        dict(type='BalanceCrossEntropyLoss', negative_ratio=3.0, eps=1e-6, pred_map='binary_map',
             target='gt', loss_name='bce', loss_weight=5),
        dict(type='MaskL1Loss', pred_map='thresh_map', target='border',
             loss_weight=10, loss_name='mask l1'),
        dict(type='DiceLoss', eps=1e-6, pred_map='thresh_binary_map', target='gt',
             loss_weight=1, loss_name='dice loss'),
    ],
    optimizer=dict(
        type='SGD',
        lr=0.007,
        momentum=0.9,
        weight_decay=1e-4,
    ),
    lr_scheduler=dict(
        type='PolyLR',
        iter_based=False,
        # milestones=[2000],
        warmup_epochs=2,
    ),
    postprocess=dict(type='Postprocessor',
                     thresh=0.3,
                     box_thresh=0.5,
                     name='binary_map',
                     min_size=8,
                     unclip_ratio=1.5,
                     ),
    max_epochs=max_epoch,
    trainval_ratio=500,
    log_interval=1,
    snapshot_interval=125,
    save_best=True,
    resume=None,
)
