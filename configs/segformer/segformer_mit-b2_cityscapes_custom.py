_base_ = ['./segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

## perturbing cityscapes ## my custom way 
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='CityTransform'),
            dict(type='Collect', keys=['img']),
        ])
]

## perturbing cityscapes ## the way segformerb2 was doing in its training part
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# crop_size = (1024, 1024)
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(2048, 1024), 
#         flip=False,
#         transforms=[
#             # dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
#             # dict(type='RandomFlip', prob=0.5),
#             # dict(type='PhotoMetricDistortion'),
#             dict(type='Resize', keep_ratio=False),
#             dict(type='PhotoMetricDistortion'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='CityTransform'),
#             dict(type='Collect', keys=['img']),
#         ])
# ]

data = dict(
    test=dict(pipeline=test_pipeline, 
            ## using 'train' folder for generating perturbed cityscapes segformerb2 prediction
            img_dir='leftImg8bit/train', 
            ann_dir='gtFine/train')
)
