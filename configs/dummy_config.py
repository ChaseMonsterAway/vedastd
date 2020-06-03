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


model = dict(
    type='GModel',
    backbone=dict(
      type='GResNet',
    ),
    enhance=dict(
      type='JunctionBlock',

    ),
    head=dict(
        type='DBHead',
    )
)