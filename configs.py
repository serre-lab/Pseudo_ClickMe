class Configs(object):

    # String Parameters
    data_dir = '/mnt/disks/clickme-with-pseudo/test'
    train_pseudo_paths = 'PseudoClickMe/train/*.pth' 
    train_clickme_paths = 'ClickMe/train/*.pth'
    val_pseudo_paths = 'PseudoClickMe/val/*.pth'
    val_clickme_paths = 'ClickMe/val/*.pth' 
    test_clickme_paths = 'ClickMe/test/*.pth' 
    model_name = "resnet50"
    weights = '/mnt/disks/bucket/pseudo_clickme/' # "/clickme-with-pseudo/ckpt/"
   
    mode = "imagenet" # "mix", "pseudo" 

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
    ckpt_remain = 5
    logger_update = 50

    # Boolean Parameters
    evaluate = False # set resume to be True at the same time
    pretrained = False
    resume = False
    tpu = True
    wandb = False
    
    @classmethod
    def set_configs(cls, args):
        if args.data_dir: cls.data_dir = args.data_dir
        if args.weights: cls.weights = args.weights
        if args.model_name: cls.model_name = args.model_name
        if args.mode: cls.mode = args.mode
        if args.epochs: cls.epochs = args.epochs
        if args.batch_size: cls.batch_size = args.batch_size
        if args.learning_rate: cls.learning_rate = args.learning_rate
        if args.interval: cls.interval = args.interval
        if args.logger_update: cls.logger_update = args.logger_update
        if args.evaluate: cls.evaluate = args.evaluate
        if args.resume: cls.resume = args.resume
        if args.pretrained: cls.pretrained = args.pretrained
        if args.tpu: cls.tpu = args.tpu
        if args.wandb: cls.wandb = args.wandb