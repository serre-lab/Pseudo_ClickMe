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
        cls.data_dir = args.get('data_dir', cls.data_dir)  
        cls.weights = args.get('weights', cls.weights)  
        cls.model_name = args.get('model_name', cls.model_name)  
        cls.mode = args.get('mode', cls.mode)  
        cls.epochs = args.get('epochs', cls.epochs)  
        cls.batch_size = args.get('batch_size', cls.batch_size)  
        cls.learning_rate = args.get('learning_rate', cls.learning_rate) 
        cls.interval = args.get('interval', cls.interval) 
        cls.logger_update = args.get('logger_update', cls.logger_update) 
        cls.evaluate = args.get('evaluate', cls.evaluate) 
        cls.resume = args.get('resume', cls.resume) 
        cls.pretrained = args.get('pretrained', cls.pretrained) 
        cls.tpu = args.get('tpu', cls.tpu) 
        cls.wandb = args.get('wandb', cls.wandb) 