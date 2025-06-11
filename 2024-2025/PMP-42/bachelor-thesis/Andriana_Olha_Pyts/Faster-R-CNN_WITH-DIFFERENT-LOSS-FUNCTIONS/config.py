RPN_config = {
    'anchor_scale': (128, 256, 512),
    'anchor_aspect_ratio': (0.5, 1.0, 2.0),
    'downsample': 16,
    'in_channels': 512,
    'num_anchors': 9,
    'bbox_reg_weights': (1.0, 1.0, 1.0, 1.0),
    'iou_positive_thresh': 0.7,
    'iou_negative_high': 0.3,
    'iou_negative_low': 0.0,
    'batch_size_per_image': 256,
    'positive_fraction': 0.5,
    'min_size': 16,
    'nms_thresh': 0.7,
    'top_n_train': 2000,
    'top_n_test': 1000
}

FastRCNN_config = {
    'output_size': 7, 
    'downsample': 16,
    'out_channels': 4096, 
    'num_classes': 21,
    'bbox_reg_weights': (10., 10., 5., 5.),
    'iou_positive_thresh': 0.5, 
    'iou_negative_high': 0.5, 
    'iou_negative_low': 0.1,
    'batch_size_per_image': 128, 
    'positive_fraction': 0.25,
    'min_size': 1, 
    'nms_thresh': 0.3,
    'score_thresh': 0.05, 
    'top_n': 50,
    'loss_type': 'focal_ciou'  
}

TRAIN_config = {
    'lr': 0.001,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'milestones': [8, 11],
    'epochs': 12,
    'print_freq': 100,
    'epoch_freq': 1,
    'save': True,
    'SAVE_PATH': './saved_models/'
}

TEST_config = {
    'num_classes': 21,
    'iou_thresh': 0.5,
    'use_07_metric': False
}

DEMO_config = {
    'min_size': 600,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'score_thresh': 0.7
}