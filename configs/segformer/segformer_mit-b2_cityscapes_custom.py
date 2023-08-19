_base_ = ['./segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_layers=[3, 4, 6, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

## perturbing cityscapes
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
data = dict(
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
