
class DefaultConfigs(object):

    # String Parameters
    data_dir = '/mnt/disks/clickme-with-pseudo/test'
    train_pseudo_paths = 'PseudoClickMe/train/*.pth' 
    train_clickme_paths = 'ClickMe/train/*.pth'
    val_pseudo_paths = 'PseudoClickMe/val/*.pth'
    val_clickme_paths = 'ClickMe/val/*.pth' 
    test_clickme_paths = 'ClickMe/test/*.pth' 
    model_name = "resnet50"
    weights = "/mnt/disks/bucket/pseudo_clickme/" # "/clickme-with-pseudo/ckpt/"
   
    # mode = "pseudo" 
    # mode = "mix"
    mode = "imagenet"

    # Numeric Parameters
    epochs = 5
    start_epoch = 0
    batch_size = 32
    momentum = 0.9
    lr = 0.1
    weight_decay = 1e-5
    interval = 10
    num_workers = 8
    step_size = 30
    gamma = 0.1
    gpu_id = 1

    # Boolean Parameters
    evaluate = False # set resume to be True at the same time
    pretrained = False
    resume = False
    tpu = True
    wandb = False