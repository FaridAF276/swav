from main_swav import *
parser = argparse.ArgumentParser(description="Implementation of lr find")
#########################
#### lr parameters ####
#########################
parser.add_argument('--lr_min', '--learning-rate', default=0.06, type=float, help='initial learning rate')
parser.add_argument('--lr_max', default=0.5, type=float, help='width of the array of lr to be tested')
parser.add_argument('--logspace', action="store_true", help='If true value between lr min and lr max will be evenly spaced in log scale')
#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", default=3000, type=int,
                    help="number of prototypes")
parser.add_argument("--queue_length", type=int, default=0,
                    help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
                    help="from this epoch, we start using a queue")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### dist parameters ###
#########################
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up distributed
                    training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--world_size", default=-1, type=int, help="""
                    number of processes: it is set automatically and
                    should not be passed as argument""")
parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                    it is set automatically and should not be passed as argument""")
parser.add_argument("--local_rank", default=0, type=int,
                    help="this argument is not used and should be ignored")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="resnet50", type=str, help="convnet architecture")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=10, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=25,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
parser.add_argument("--sync_bn", type=str, default="pytorch", help="synchronize bn")
parser.add_argument("--syncbn_process_group_size", type=int, default=8, help=""" see
                    https://github.com/NVIDIA/apex/blob/master/apex/parallel/__init__.py#L58-L67""")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")

def main():
    lr_values = np.logspace(args.lr_min, args.lr_max, num= 10) if args.logspace else np.linspace(args.lr_min, args.lr_max, num= 10) 
    global args
    args = parser.parse_args()
    init_distributed_mode(args)
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # build data
    train_dataset = MultiCropDataset(
        args.data_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info("Building data done with {} images loaded.".format(len(train_dataset)))
    for lr_item in lr_values : 
        args.base_lr=lr_item
         # build model
        model = resnet_models.__dict__[args.arch](
            normalize=True,
            hidden_mlp=args.hidden_mlp,
            output_dim=args.feat_dim,
            nmb_prototypes=args.nmb_prototypes,
        )
        # synchronize batch norm layers
        if args.sync_bn == "pytorch":
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        elif args.sync_bn == "apex":
            # with apex syncbn we sync bn per group because it speeds up computation
            # compared to global syncbn
            process_group = apex.parallel.create_syncbn_process_group(args.syncbn_process_group_size)
            model = apex.parallel.convert_syncbn_model(model, process_group=process_group)
        # copy model to GPU
        model = model.cuda()
        if args.rank == 0:
            logger.info(model)
        logger.info("Building model done.")

        # build optimizer
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.base_lr,
            momentum=0.9,
            weight_decay=args.wd,
        )
        optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
        warmup_lr_schedule = np.linspace(args.start_warmup, args.base_lr, len(train_loader) * args.warmup_epochs)
        iters = np.arange(len(train_loader) * (args.epochs - args.warmup_epochs))
        cosine_lr_schedule = np.array([args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (1 + \
                            math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs)))) for t in iters])
        lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        logger.info("Building optimizer done.")

        # init mixed precision
        if args.use_fp16:
            model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
            logger.info("Initializing mixed precision done.")

        # wrap model
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu_to_work_on]
        )

        # optionally resume from a checkpoint
        to_restore = {"epoch": 0}
        restart_from_checkpoint(
            os.path.join(args.dump_path, "checkpoint.pth.tar"),
            run_variables=to_restore,
            state_dict=model,
            optimizer=optimizer,
            amp=apex.amp,
        )
        start_epoch = to_restore["epoch"]

        # build the queue
        queue = None
        queue_path = os.path.join(args.dump_path, "queue" + str(args.rank) + ".pth")
        if os.path.isfile(queue_path):
            queue = torch.load(queue_path)["queue"]
        # the queue needs to be divisible by the batch size
        args.queue_length -= args.queue_length % (args.batch_size * args.world_size)

        cudnn.benchmark = True

        for epoch in range(start_epoch, args.epochs):

            # train the network for one epoch
            logger.info("============ Starting epoch %i ... ============" % epoch)

            # set sampler
            train_loader.sampler.set_epoch(epoch)

            # optionally starts a queue
            if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
                queue = torch.zeros(
                    len(args.crops_for_assign),
                    args.queue_length // args.world_size,
                    args.feat_dim,
                ).cuda()

            # train the network
            scores, queue = train(train_loader, model, optimizer, epoch, lr_schedule, queue)
            training_stats.update(scores)

            # save checkpoints
            if args.rank == 0:
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if args.use_fp16:
                    save_dict["amp"] = apex.amp.state_dict()
                torch.save(
                    save_dict,
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                )
                if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                    shutil.copyfile(
                        os.path.join(args.dump_path, "checkpoint.pth.tar"),
                        os.path.join(args.dump_checkpoints, "ckp-" + str(epoch) + ".pth"),
                    )
            if queue is not None:
                torch.save({"queue": queue}, queue_path)
   
if __name__ == "__main__":
    main()
