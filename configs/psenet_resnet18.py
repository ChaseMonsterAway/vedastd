# config for inference

norm_cfg = dict(type='BN')
kernel_num = 7

# 9. device
inference = dict(
    gpu_id='0',
    transforms=[
        dict(type='Resize', keep_ratio=True, size=(640, 640),
             img_mode='nearest'),
        dict(type='PadIfNeeded', factor=32, pad_value=0),
        dict(type='FilterKeys',
             need_keys=['polygon', 'tags', 'shape', 'input', 'ratio']),
        dict(type='ToTensor', keys=['input']),
        dict(type='Normalize', mean=(123.675, 116.280, 103.530),
             std=(58.395, 57.120, 57.375), key='input'),
    ],
    postprocessor=dict(
        type='PsePostprocessor',
        debug=False,
        resize=True,
        thresh=1.0,
        min_kernel_area=5,
        max_candidates=10,
        min_area=50,
        name=('pred_text_map', 'pred_kernels_map'),
    ),
    model=dict(
        type='PseNet',
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
)
# 2. configuration for train/test
root_workdir = 'workdir/'
dataset_type = 'TxtDataset'
dataset_root = 'datasets/icdar2015/'

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
                      img_root=dataset_root + 'train_images',
                      gt_root=dataset_root + 'train_gts',
                      txt_file=dataset_root + 'train_list.txt',
                      )],
        transforms=inference['transforms'],
        dataloader=dict(type='BaseDataloader',
                        batch_size=1)
    )
)
## 2.2 configuration for train
max_epoch = 100
max_iterations = 200
train = dict(
    data=dict(
        train=dict(
            dataset=[dict(type=dataset_type,
                          img_root=dataset_root + 'train_images',
                          gt_root=dataset_root + 'train_gts',
                          txt_file=dataset_root + 'train_list.txt',
                          )],
            transforms=[
                dict(type='MakeShrinkMap', ratios=[1.0], max_shr=20,
                     min_text_size=8,
                     prefix='text'),
                dict(type='MakeShrinkMap',
                     ratios=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                     max_shr=20, min_text_size=8, prefix='kernels'),
                dict(type='Resize', keep_ratio=True, size=(640, 640),
                     img_mode='nearest'),
                dict(type='PadIfNeeded', factor=32, pad_value=0),
                # dict(type='Resize', keep_ratio=False, random_scale=True, size=(640, 640),
                #      img_mode='cubic', mask_mode='nearest', max_size=1280,
                #      scale_list=([0.5, 1.0, 2.0, 3.0])),
                # dict(type='RandomFlip', p=0.5, horizontal=True, vertical=False),
                # dict(type='RandomRotation', angles=(-10, 10), p=1),
                # dict(type='RandomCrop', size=(640, 640), prefix='text'),
                # dict(type='FilterKeys',
                #     need_keys=['shape', 'input', 'text_map', 'text_mask', 'kernels_map',
                #                'kernels_mask']),
                dict(type='ToTensor',
                     keys=['input', 'text_map', 'text_mask', 'kernels_map',
                           'kernels_mask']),
                # input value are 0-1
                # dict(type='Normalize', mean=[0.485, 0.456, 0.406],
                #      std=[0.229, 0.224, 0.225], key='input'),
                # input value are 0-255
                dict(type='Normalize', mean=(123.675, 116.280, 103.530),
                     std=(58.395, 57.120, 57.375), key='input'),
            ],
            dataloader=dict(type='BaseDataloader', batch_size=1),
        ),
        val=dict(
            dataset=[dict(type=dataset_type,
                          img_root=dataset_root + 'train_images',
                          gt_root=dataset_root + 'train_gts',
                          txt_file=dataset_root + 'train_list.txt',
                          )],
            transforms=inference['transforms'],
            dataloader=dict(type='BaseDataloader', batch_size=1)
        ),
    ),
    criterion=[
        dict(type='DiceLoss', eps=1e-6, pred_map='pred_text_map',
             gt_map='text_map',
             gt_mask='text_mask',
             loss_weight=0.7, loss_name='text dice loss', ohem=False),
        dict(type='MultiDiceLoss', eps=1e-6, score_map='pred_text_map',
             pred_map='pred_kernels_map', gt_map='kernels_map',
             gt_mask='text_mask', loss_weight=0.3,
             loss_name='kernels dice loss',
             ohem=False),
    ],
    optimizer=dict(
        type='SGD',
        lr=0.001,
        momentum=0.99,
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
    snapshot_interval=5,
    save_best=True,
)
