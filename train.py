import os
import torch
import datetime

from darknet import *
#testing
from datasets.pascal_voc import VOCDataset
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg
from random import randint
from test import *

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None

def myloss(bbox_pred, iou_pred, prob_pred, box_mask, iou_mask, class_mask, _boxes, _ious, _classes, num_boxes):
    bbox_loss = nn.MSELoss(size_average=False)(bbox_pred * box_mask, _boxes * box_mask) / num_boxes  # noqa
    iou_loss = nn.MSELoss(size_average=False)(iou_pred * iou_mask, _ious * iou_mask) / num_boxes  # noqa
    cls_loss = nn.MSELoss(size_average=False)(prob_pred * class_mask, _classes * class_mask) / num_boxes  # n
    return bbox_loss,iou_loss,cls_loss,bbox_loss+iou_loss+cls_loss

# data loader
imdb = VOCDataset(cfg.imdb_train, cfg.DATA_DIR, cfg.train_batch_size,
                  yolo_utils.preprocess_train, processes=2, shuffle=True,
                  dst_size=cfg.multi_scale_inp_size)
# dst_size=cfg.inp_size)
print('load data succ...')
net = YOLOPCN(cls =0)
#net = Darknet19()
# net_utils.load_net(cfg.trained_model, net)
# pretrained_model = os.path.join(cfg.train_output_dir,
#     'darknet19_voc07trainval_exp1_63.h5')
# pretrained_model = cfg.trained_model
# net_utils.load_net(pretrained_model, net)
#net.load_from_npz(cfg.pretrained_model, num_conv=18)
#net.cuda()
net = torch.nn.DataParallel(net).cuda()
net.train()

# optimizer
start_epoch = 0
lr = cfg.init_learning_rate
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay)

# tensorboad
use_tensorboard = cfg.use_tensorboard and SummaryWriter is not None
# use_tensorboard = False
if use_tensorboard:
    summary_writer = SummaryWriter(os.path.join(cfg.TRAIN_DIR, 'runs', cfg.exp_name))
else:
    summary_writer = None

batch_per_epoch = imdb.batch_per_epoch
train_loss = 0
bbox_loss, iou_loss, cls_loss = 0., 0., 0.
cnt = 0
t = Timer()
step_cnt = 0
size_index = 0

imdb_name = cfg.imdb_test
imdb2 = VOCDataset(imdb_name, cfg.DATA_DIR, cfg.batch_size,
                      yolo_utils.preprocess_test,
                      processes=1, shuffle=False, dst_size=cfg.multi_scale_inp_size)
max_per_image = 300
thresh = 0.01
vis = False
output_dir = cfg.test_output_dir

bbox_loss_sum = 0.0
iou_loss_sum = 0.0
cls_loss_sum = 0.0     #cls = conditional class probability, prob detected object belong to class
train_loss_sum = 0.0

for step in range(start_epoch * imdb.batch_per_epoch,
                  cfg.max_epoch * imdb.batch_per_epoch):
    t.tic()
    # batch
    batch = imdb.next_batch(size_index)
    im = batch['images']
    gt_boxes = batch['gt_boxes']
    gt_classes = batch['gt_classes']
    dontcare = batch['dontcare']
    orgin_im = batch['origin_im']

    num_boxes = sum((len(boxes) for boxes in gt_boxes))

    # forward
    im_data = net_utils.np_to_variable(im,
                                       is_cuda=True,
                                       volatile=False).permute(0, 3, 1, 2)
 #   bbox_pred, iou_pred, prob_pred = net(im_data, gt_boxes, gt_classes, dontcare, size_index)
    bbox_pred, iou_pred, prob_pred, box_mask, iou_mask, class_mask, _boxes, _ious, _classes  = net(im_data, gt_boxes, gt_classes, dontcare, size_index)
    # backward
 #   loss = net.loss
    bbox_loss,iou_loss,cls_loss,loss = myloss(bbox_pred, iou_pred, prob_pred, box_mask, iou_mask, class_mask,_boxes, _ious, _classes, num_boxes)
#    print(bbox_loss.data,net.bbox_loss.data)
  #  print(bbox_loss,'iouloss: ',iou_loss,' clsloss:',cls_loss,' trainloss:',loss)

    bbox_loss_sum += float(bbox_loss.data.cpu().numpy())
    iou_loss_sum += float(iou_loss.data.cpu().numpy())
    cls_loss_sum += float(cls_loss.data.cpu().numpy())
    train_loss_sum += float(loss.data.cpu().numpy())
        #print(bbox_loss,'iouloss: ',iou_loss,' clsloss:',cls_loss,' trainloss:',train_loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    cnt += 1
    step_cnt += 1
    duration = t.toc()
    if step % cfg.disp_interval == 0:
        train_loss_sum /= cnt
        bbox_loss_sum /= cnt
        iou_loss_sum /= cnt
        cls_loss_sum /= cnt
        print(('epoch %d[%d/%d], loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, '
               'cls_loss: %.3f (%.2f s/batch, rest:%s)' %
               (imdb.epoch, step_cnt, batch_per_epoch, train_loss_sum, bbox_loss_sum,
                iou_loss_sum, cls_loss_sum, duration,
                str(datetime.timedelta(seconds=int((batch_per_epoch - step_cnt) * duration))))))  # noqa

        if summary_writer and step % cfg.log_interval == 0:
            summary_writer.add_scalar('loss_train', train_loss_sum, step)
            summary_writer.add_scalar('loss_bbox', bbox_loss_sum, step)
            summary_writer.add_scalar('loss_iou', iou_loss_sum, step)
            summary_writer.add_scalar('loss_cls', cls_loss_sum, step)
            summary_writer.add_scalar('learning_rate', lr, step)

        bbox_loss_sum = 0.0
        iou_loss_sum = 0.0
        cls_loss_sum = 0.0
        train_loss_sum = 0.0
        cnt = 0
        t.clear()
        size_index = randint(0, len(cfg.multi_scale_inp_size) - 1)
        print("image_size {}".format(cfg.multi_scale_inp_size[size_index]))

    if step > 0 and (step % imdb.batch_per_epoch == 0):
        if imdb.epoch in cfg.lr_decay_epochs:
            lr *= cfg.lr_decay
            optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                        momentum=cfg.momentum,
                                        weight_decay=cfg.weight_decay)

        save_name = os.path.join(cfg.train_output_dir,
                                 '{}_{}.h5'.format(cfg.exp_name, imdb.epoch))
        net_utils.save_net(save_name, net)
        print(('save model: {}'.format(save_name)))
        step_cnt = 0
   
#    test_net(net, imdb2, max_per_image, thresh, vis)

