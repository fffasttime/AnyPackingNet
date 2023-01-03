import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time

import sys
sys.path.append('..')
import localconfig
import test
from datasets import *
from yolo_utils import *

from mymodel import *
import mymodel

wdir = 'weights' + os.sep  # weights dir

# Hyperparameters (results68: 59.9 mAP@0.5 yolov3-spp-416) https://github.com/ultralytics/yolov3/issues/310

hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.5,  # focal loss gamma
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98,  # image rotation (+/- deg)
       'translate': 0.05,  # image translation (+/- fraction)
       'scale': 0.05,  # image scale (+/- gain)
       'shear': 0.641}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v

def train():
    img_size, img_size_test = opt.img_size if len(opt.img_size) == 2 else opt.img_size * 2  # train, test sizes
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights  # initial training weights

    # Initialize
    init_seeds()

    # Configure run
    train_path = localconfig.train_path
    test_path = localconfig.test_path
    nc = 1 

    results_file = 'results/%s.txt'%opt.name

    # Initialize model
    if opt.model != '':
        model = getattr(mymodel, opt.model)(opt.bitw, opt.bita).to(device)
    else:
        if opt.bypass:
            model = UltraNetBypass_FixQ(opt.bitw, opt.bita).to(device)
        else:
            model = UltraNet_FixQ(opt.bitw, opt.bita).to(device)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if opt.adam:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    optimizer.param_groups[2]['lr'] *= 2.0  # bias lr

    del pg0, pg1, pg2

    start_epoch = 0
    test_best_iou = 0.0

    # load weights
    if weights.endswith('.pt'):  # pytorch format
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        chkpt = torch.load(weights, map_location=device)

        # load model
        try:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        if opt.resume:
        # load optimizer
            if chkpt['optimizer'] is not None:
                optimizer.load_state_dict(chkpt['optimizer'])
                best_fitness = chkpt['best_fitness']

            # load results
            if chkpt.get('training_results') is not None:
                with open(results_file, 'w') as file:
                    file.write(chkpt['training_results'])  # write results.txt

            start_epoch = chkpt['epoch'] + 1

        del chkpt

    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    lf = lambda x: (1 + math.cos(x * math.pi / epochs)) / 2 * 0.999 + 0.001  # cosine https://arxiv.org/pdf/1812.01187.pdf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:5000',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataloader
    #batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count()//4, batch_size//4 if batch_size > 1 else 0, 8])  # number of workers

    # Testloader
    testset = LoadImagesAndLabels(test_path, img_size_test, batch_size,
                                                                 hyp=hyp,
                                                                 rect=False,
                                                                 cache_images=opt.cache_images,
                                                                 single_cls=opt.single_cls)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             num_workers=0,
                                             pin_memory=True,
                                             collate_fn=testset.collate_fn)

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  cache_images=opt.cache_images,
                                  single_cls=opt.single_cls)
                                             
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Start training
    nb = len(dataloader)
    prebias = start_epoch == 0
    model.nc = nc  # attach number of classes to model
    model.arc = opt.arc  # attach yolo architecture
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    torch_utils.model_info(model, report='summary')  # 'full' or 'summary'
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)

    test.test(batch_size=batch_size,
                                img_size=img_size_test,
                                model=model,
                                dataloader=testloader) # make forward
    bops, bita, bitw, dsps = model.fetch_arch_info()
    print('model with bops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M, dsps: {:.3f}M'.format(bops, bita, bitw, dsps))
            
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        model.gr = 1 - (1 + math.cos(min(epoch * 2, epochs) * math.pi / epochs)) / 2  # GIoU <-> 1.0 loss ratio

        # Prebias
        if prebias:
            ne = max(round(30 / nb), 3)  # number of prebias epochs
            ps = np.interp(epoch, [0, ne], [0.1, hyp['lr0'] * 2]), \
                 np.interp(epoch, [0, ne], [0.9, hyp['momentum']])  # prebias settings (lr=0.1, momentum=0.9)
            if epoch == ne:
                # print_model_biases(model)
                prebias = False

            # Bias optimizer settings
            optimizer.param_groups[2]['lr'] = ps[0]
            if optimizer.param_groups[2].get('momentum') is not None:  # for SGD but not Adam
                optimizer.param_groups[2]['momentum'] = ps[1]

        mloss = torch.zeros(4).to(device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'iouloss', 'objloss', 'triou', 'mloss', 'targets', 'img_size'))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 256.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)

            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Scale loss by nominal batch_size of 64
            loss *= batch_size / 64

            loss.backward()

            # Optimize accumulated gradient
            if ni % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

            # end batch ------------------------------------------------------------------------------------------------
        
        # Update scheduler
        scheduler.step()

        # Process epoch results
        final_epoch = epoch + 1 == epochs
        if not opt.notest or final_epoch:  # Calculate mAP
            results = test.test(batch_size=batch_size,
                                img_size=img_size_test,
                                model=model,
                                dataloader=testloader)
        
        # Write epoch results
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * len(results) % results + '\n')  # test_losses=(iou, loss_sum, lobj, lcls)

        # Update best mAP
        results =  torch.tensor(results, device = 'cpu')
               
        test_iou = results[0]
        if test_iou > test_best_iou:
            test_best_iou = test_iou

        # Save training results
        save = (not opt.nosave) or (final_epoch)
        if save:
            with open(results_file, 'r') as f:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'training_results': f.read(),
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict(),
                         'model_params':model.model_params, # arch param
                         'extra': {'time': time.ctime(), 'name': opt.name}}

            # Save last checkpoint
            torch.save(chkpt, wdir + '%s_last.pt'%opt.name)
            
            if test_iou == test_best_iou:
                torch.save(chkpt, wdir + '%s_best.pt'%opt.name)

            # Delete checkpoint
            del chkpt

        # end epoch ----------------------------------------------------------------------------------------------------

    # end training
    n = opt.name

    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bypass', action='store_true', help='use bypass model')
    parser.add_argument('--epochs', type=int, default=200)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=64)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--accumulate', type=int, default=1, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny-1cls_1.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2017.data', help='*.data path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[320], help='train and test image-sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--arc', type=str, default='default', help='yolo architecture')  # default, uCE, uBCE
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--bitw', type=str, default='')
    parser.add_argument('--bita', type=str, default='')
    parser.add_argument('--var', type=float, help='debug variable')
    parser.add_argument('--model', type=str, default='', help='use specific model')
  
    opt = parser.parse_args()
    last = wdir + 'last_%s.pt'%opt.name
    opt.weights = last if opt.resume else opt.weights
    print(opt)
    device = torch_utils.select_device(opt.device, batch_size=opt.batch_size)

    train()  # train normally
