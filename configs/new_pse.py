# config for inference

norm_cfg = dict(type='BN')
kernel_num = 7

# 9. device
deploy = dict(
    gpu_id='0',
    transforms=[
        dict(type='LongestMaxSize', max_size=512, interpolation='nearest', p=1),
        dict(type='PadIfNeeded', min_height=512, min_width=640, border_mode='constant',
             value=0),
        dict(type='Normalize', mean=(123.675, 116.280, 103.530),
             std=(58.395, 57.120, 57.375), max_pixel_value=255),
        dict(type='ToTensor'),
    ],
    postprocessor=dict(
        type='PsePostprocessor',
        debug=False,
        resize=True,
        thresh=0.5,
        min_kernel_area=5,
        max_candidates=10,
        min_area=50,
        name=('pred_text_map', 'pred_kernels_map'),
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
                                      ),
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
                                      ),
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
                                      align_corners=True,
                                      ),
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
    metric=dict(type='QuadMeasurer', polygon=False, box_thresh=0.6),
)

## 2.1 configuration for test
test = dict(
    data=dict(
        dataset=[dict(type=dataset_type,
                      img_root=dataset_root + r'\ch4_training_images',
                      gt_root=dataset_root + r'\ch4_training_localization_transcription_gt',
                      txt_file=dataset_root + r'\train.txt',
                      )],
        transforms=deploy['transforms'],
        dataloader=dict(type='BaseDataloader',
                        batch_size=1,
                        )
    )
)
## 2.2 configuration for train
max_epoch = 100
max_iterations = 200
train = dict(
    data=dict(
        train=dict(
            dataset=[dict(type=dataset_type,
                          img_root=dataset_root + r'\ch4_training_images',
                          gt_root=dataset_root + r'\ch4_training_localization_transcription_gt',
                          txt_file=dataset_root + r'\train.txt',
                          )],
            transforms=[
                dict(type='MakeShrinkMap', ratios=[1.0], max_shr=20, min_text_size=8, p=1),
                dict(type='MaskMarker', name='gt'),
                dict(type='MakeShrinkMap', ratios=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                     max_shr=20, min_text_size=8, p=1),
                dict(type='MaskMarker', name='shrink'),
                dict(type='LongestMaxSize', max_size=512, interpolation='bilinear', p=1),
                dict(type='PadIfNeeded', min_height=512, min_width=512, border_mode='constant',
                     value=0),
                dict(type='Normalize', mean=(123.675, 116.280, 103.530),
                     std=(58.395, 57.120, 57.375), max_pixel_value=255),
                dict(type='FilterKeys', op_names=['tags', 'each_len']),
                dict(type='ToTensor'),
                dict(type='Grouping'),
            ],
            collate_fn=dict(type='BaseCollate', stack_keys=['image', 'gt', 'shrink']),
            dataloader=dict(type='BaseDataloader', batch_size=2),
        ),
        val=dict(
            dataset=[dict(type=dataset_type,
                          img_root=dataset_root + r'\ch4_training_images',
                          gt_root=dataset_root + r'\ch4_training_localization_transcription_gt',
                          txt_file=dataset_root + r'\train.txt',
                          )],
            transforms=deploy['transforms'],
            dataloader=dict(type='BaseDataloader', batch_size=2)
        ),
    ),
    criterion=[
        dict(type='DiceLoss', eps=1e-6,
             pred_map='pred_text_map',
             target='gt',
             loss_weight=0.7,
             loss_name='text dice loss',
             ohem=False),
        dict(type='MultiDiceLoss',
             eps=1e-6,
             score_map='pred_text_map',
             pred_map='pred_kernels_map',
             target='shrink',
             loss_weight=0.3,
             loss_name='kernels dice loss',
             ohem=False),
    ],
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
    ),
    lr_scheduler=dict(
        type='StepLR',
        milestones=[60, 120],
        gamma=0.1),
    resume=None,
    max_iterations=max_iterations,
    max_epochs=max_epoch,
    trainval_ratio=10,
    log_interval=1,
    snapshot_interval=20,
    save_best=True,
)
