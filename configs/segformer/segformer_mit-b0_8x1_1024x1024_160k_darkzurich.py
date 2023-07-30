_base_ = [
    '../_base_/models/segformer_mit-b0.py',
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=checkpoint)),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))



dataset_type = 'DarkZurichDataset'
data_root = '/home/sidd_s/scratch/dataset/DarkZurich' 
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='DarkZurichDataset',
        data_root='/home/sidd_s/scratch/dataset/DarkZurich' ,
        img_dir='rgb_anon/train',
        ann_dir='gt/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotationsCityscapes20'),
            dict(
                type='Resize', img_scale=(1920,1080), ratio_range=(0.5, 2.0)),
            dict(
                type='RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(1024, 1024), pad_val=0, seg_pad_val=0),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='Cityscapes20Dataset',
        data_root='/home/sidd_s/scratch/dataset/DarkZurich' ,
        img_dir='rgb_anon/val',
        ann_dir='gt/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1920,1080),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='DarkZurichDataset',
        data_root='/home/sidd_s/scratch/dataset/DarkZurich' ,
        img_dir='rgb_anon/val',
        ann_dir='gt/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1920,1080),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000, max_keep_ckpts=1)
evaluation = dict(
    interval=16000, metric='mIoU', pre_eval=True, save_best='mIoU')

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)