import argparse

from torch.utils.data import DataLoader

import sys
sys.path.append('..')
from datasets import *
from yolo_utils import *

import mymodel
from mymodel import *
from utils.view_pt import select_weight_file
import cv2

opt=None

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


def save_test_pic(filename, img, pbox, tbox):
    img=img.numpy().transpose((1,2,0))*255
    img=np.ascontiguousarray(img)

    pp1, pp2 = (int(pbox[0]-pbox[2]/2), int(pbox[1]-pbox[3]/2)), (int(pbox[0]+pbox[2]/2), int(pbox[1]+pbox[3]/2))
    tp1, tp2 = (int(tbox[0]-tbox[2]/2), int(tbox[1]-tbox[3]/2)), (int(tbox[0]+tbox[2]/2), int(tbox[1]+tbox[3]/2))

    cv2.rectangle(img, pp1, pp2, color=(0,0,255), thickness=1) # red pbox
    cv2.rectangle(img, tp1, tp2, color=(0,255,0), thickness=1) # green tbox
    cv2.putText(img, text=str((pp1,pp2))+str((tp1, tp2)),
                org = (0, 10),
                fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                fontScale=0.35,
                color = (255,255,255))

    cv2.imwrite('test_result/'+filename+'.jpg', img)

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
 
    # Transform from center and width to exact coordinates
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def select_boxes(pred_boxes, pred_conf):
    n = pred_boxes.size(0)
    # pred_boxes = pred_boxes.view(n, -1, 4)
    # pred_conf = pred_conf.view(n, -1, 1)
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
    p_boxes = FloatTensor(n, 4)
    # print(pred_boxes.shape, pred_conf.shape)

    for i in range(n):
        _, index = pred_conf[i].max(0)
        p_boxes[i] = pred_boxes[i][index]

    return p_boxes

def get_prebox(inf_out):
    inf_out = inf_out.view(inf_out.shape[0], 6, -1) # bs, anchors, nw*nh*6
    inf_out_t = torch.zeros_like(inf_out[:, 0, :])
    for i in range(inf_out.shape[1]):
        inf_out_t += inf_out[:, i, :]
    inf_out_t = inf_out_t.view(inf_out_t.shape[0], -1, 6) / 6 # average anchors: box, conf

    pre_box = select_boxes(inf_out_t[..., :4], inf_out_t[..., 4]) # get pbox by max conf
    return pre_box

def test(weights=None,
         batch_size=16,
         img_size=416,
         model=None,
         dataloader=None,
         num_batch=-1):
    # Initialize/load model and set device
    if model is None or type(model)==str:
        device = torch_utils.select_device(opt.device, batch_size=batch_size)

        # Remove previous
        for f in glob.glob('test_batch*.jpg'):
            os.remove(f)

        ptfile: Dict = torch.load('weights/' + weights+'.pt', map_location=device)
        model_params = ptfile.setdefault('model_params')
        print('model_params', model_params)
        model = getattr(mymodel, model)(**model_params).to(device)

        model.hyp = hyp
        model.nc = 1
        model.arc = 'default'

        # Load weights
        model.load_state_dict(ptfile['model'])

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:  # called by train.py
        device = next(model.parameters()).device  # get model device
    
    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(opt.datapath, img_size, batch_size, rect=False, cache_labels=True, hyp=hyp, augment=False)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                #num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    model.eval()
    loss = torch.zeros(2)
    iou_sum = 0
    test_n = 0
    
    # model.layers[0].weight.data = torch.tensor(model.layers[0].weight.data.numpy()[:,::-1].copy()) # swap RGB<->BGR

    print(('\n' + '%10s' * 4) % ('IOU', 'l', 'Giou-l', 'obj-l'))
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))    
    for batch_i, (imgs, targets, paths, shapes) in pbar:
        if batch_i == num_batch: break

        imgs = imgs.to(device).float() / 256.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        bn, _, height, width = imgs.shape  # batch size, channels, height, width
        test_n += bn

        with torch.no_grad():
            # Run model
            inf_out, train_out = model(imgs)  # inference and training outputs, inf_out = bs*anchors*nw*nh*6
            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:2].cpu()  # GIoU, obj

            pre_box = get_prebox(inf_out) # anchor average, select max

            tbox = targets[..., 2:6] * torch.Tensor([width, height, width, height]).to(device)

            ious = bbox_iou(pre_box, tbox)
            iou_sum += ious.sum()
            loss_o = loss / (batch_i + 1)

            iou = iou_sum / test_n
            s = (('%10.4f')*4+'%10d') % (iou, loss_o.sum(), loss_o[0], loss_o[1], len(targets))


            if opt and opt.verbose:
                np.set_printoptions(precision = 2)
                for p in range(len(imgs)):
                    print(paths[p], 'pbox_xywh', pre_box[p].numpy(), 'tbox_xywh', tbox[p].numpy())

            if opt and opt.save_pic:
                for p in range(len(imgs)):
                    save_test_pic(str(p+test_n-batch_size), imgs[p], pre_box[p], tbox[p])

            pbar.set_description(s)
            
    return iou, loss_o.sum(), loss_o[0], loss_o[1] # iou, loss_sum, lobj, lcls
           

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('-m', '--model', type=str, default='UltraNet_FixQ', help='model name')
    parser.add_argument('-w', '--weight', default=None, help='weights path')
    parser.add_argument('-bs', '--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--datapath', default='../../dacsdc_dataset', help = 'test dataset path')
    parser.add_argument('--verbose', action='store_true', help = 'show predict value result')
    parser.add_argument('--save-pic', action='store_true', help = 'save predict output picture')
    parser.add_argument('-nb', '--num-batch', type=int, default='-1', help='num of batchs to run, -1 for full dataset')
    opt = parser.parse_args()
    print(opt)
    if opt.weight is None: opt.weight = select_weight_file()

    # Test
    res = test(
            opt.weight,
            opt.batch_size,
            opt.img_size,
            opt.model,
            num_batch = opt.num_batch)

    print(('%s %s.pt\niou %.4f, lsum %.4f, lobj %.4f, lcls %.4f')%(opt.model, opt.weight, *res))
