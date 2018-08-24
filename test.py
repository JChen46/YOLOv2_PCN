import os
import cv2
import numpy as np
import pickle
import argparse

from darknet import *
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
from datasets.pascal_voc import VOCDataset
import cfgs.config as cfg

def myloss(bbox_pred, iou_pred, prob_pred, box_mask, iou_mask, class_mask, _boxes, _ious, _classes, num_boxes):
    bbox_loss = nn.MSELoss(size_average=False)(bbox_pred * box_mask, _boxes * box_mask) / num_boxes  # noqa
    iou_loss = nn.MSELoss(size_average=False)(iou_pred * iou_mask, _ious * iou_mask) / num_boxes  # noqa
    cls_loss = nn.MSELoss(size_average=False)(prob_pred * class_mask, _classes * class_mask) / num_boxes  # n
    return bbox_loss,iou_loss,cls_loss,bbox_loss+iou_loss+cls_loss

parser = argparse.ArgumentParser(description='YOLO object detection with PCN')
parser.add_argument('--multi', default=False, type=bool, help='(def:False) for multi GPU processing')
parser.add_argument('--cls', default=1, type=int, help='(def:1) number of cycles')
parser.add_argument('--pretrained', default = True,type=bool, help='(def:True) loads pretrained model')
#parser.add_argument('--weightfile', default = 'checkpoint_cls1.pth.tar',type=str, help='(def:checkpoint_cls1.pth.tar) which weight file to train from')
parser.add_argument('--lr', default = 0.001, type=float, help='(def:0.001) learning rate')
parser.add_argument('--trainedfolder', default = 'this_means_nothing', type=str, help='(def:training) folder that contains the trained weight files') #this doesn't work since it is referenced in config.py, and only referenced when training
parser.add_argument('--filenum', default = 0, type=int, help='(def:0) weight file epoch number. Varies based on weight file name')
parser.add_argument('--image_size_index', type=int, default=0,
                    metavar='image_size_index',
                    help='setting images size index 0:320, 1:352, 2:384, 3:416, 4:448, 5:480, 6:512, 7:544, 8:576')
args = parser.parse_args()

cfg.set_train_directory(args.trainedfolder) #sends trained folder name to config.py

#printing out parse arguments --------------------------------------------
print('\nArguments: \n    multi: ', args.multi, '\n    cls: ', args.cls, '\n    pretrained: ', args.pretrained, '\n    lr: ', args.lr , '\n    trainedfolder: ', args.trainedfolder, '\n    filenum: ', 'darknet19_voc07trainval_exp3_{}.h5'.format(args.filenum), '\n') 

#save mAP results to file -----------------------
logpath = './logs'
#if not os.path.isdir(checkpointpath):
#    os.mkdir(checkpointpath)
if not os.path.isdir(logpath):
    os.mkdir(logpath)
mapfile = open('./logs/testMAP_{}.txt'.format(cfg.traindirectory), 'a')


# hyper-parameters
# ------------
imdb_name = cfg.imdb_test
# trained_model = cfg.trained_model
trained_file_name = 'darknet19_voc07trainval_exp3_{}.h5'.format(args.filenum) #added file name from parser
trained_model = os.path.join(cfg.train_output_dir,
                             trained_file_name)
output_dir = cfg.test_output_dir

max_per_image = 300
thresh = 0.01
vis = False
# ------------


def test_net(net, imdb, gtboxes, gtclasses, dontcare, sizeindex, max_per_image=300, thresh=0.5, vis=False):
    num_images = imdb.num_images

    #print('sizeindex is ', sizeindex)

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(output_dir, 'detections.pkl')
    size_index = args.image_size_index

    loss_sum = 0.0

    for i in range(num_images):

        batch = imdb.next_batch(size_index=size_index)
        ori_im = batch['origin_im'][0]
        im_data = net_utils.np_to_variable(batch['images'], is_cuda=True,
                                           volatile=True).permute(0, 3, 1, 2)

        _t['im_detect'].tic()

        bbox_pred, iou_pred, prob_pred = net(im_data, gtboxes, gtclasses, dontcare, sizeindex, False)
        #bbox_pred, iou_pred, prob_pred, box_mask, iou_mask, class_mask, _boxes, _ious, _classes  = net(im_data, gtboxes, gtclasses, dontcare, sizeindex)
        #print('net.loss.data: ', net.loss.data) #same as FINAL LOSS, also this prints a million times
#        if args.multi:
#            bbox_loss,iou_loss,cls_loss,loss = myloss(bbox_pred, iou_pred, prob_pred, box_mask, iou_mask, class_mask,_boxes, _ious, _classes, num_boxes) #multi-GPU
#            loss_sum += float(loss.data)
#        else:
#            loss_sum += net.loss.data

        # to numpy
        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()

        bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred,
                                                          iou_pred,
                                                          prob_pred,
                                                          ori_im.shape,
                                                          cfg,
                                                          thresh,
                                                          size_index
                                                          )
        detect_time = _t['im_detect'].toc()

        _t['misc'].tic()

        for j in range(imdb.num_classes):
            inds = np.where(cls_inds == j)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_dets = np.hstack((c_bboxes,
                                c_scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = c_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        nms_time = _t['misc'].toc()

        if i % 1000 == 0: #changed to 1000
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images, detect_time, nms_time))  # noqa
            _t['im_detect'].clear()
            _t['misc'].clear()

        if vis: 
            im2show = yolo_utils.draw_detection(ori_im,
                                                bboxes,
                                                scores,
                                                cls_inds,
                                                cfg,
                                                thr=0.1)
            if im2show.shape[0] > 1100:
                im2show = cv2.resize(im2show,
                                     (int(1000. * float(im2show.shape[1]) / im2show.shape[0]), 1000))  # noqa
            cv2.imshow('test', im2show)
            cv2.waitKey(0)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

#    final_loss = loss_sum / (num_images) #calculating testing loss

#    mapfile.write('%d,%f,%f\n' % (imdb.epoch, imdb.getmap,final_loss.item())) #epoch, map score, test loss
    mapfile.write('%d,%f\n' % (imdb.epoch, imdb.getmap))

    print('trainedfolder: ', args.trainedfolder, '\nfilenum: ', args.filenum)
#mapfile.close()
if __name__ == '__main__': #Only does this if test is directly called from cmd line
    # data loader
    imdb = VOCDataset(imdb_name, cfg.DATA_DIR, cfg.batch_size,
                      yolo_utils.preprocess_test,
                      processes=1, shuffle=False, dst_size=cfg.multi_scale_inp_size)

    net = YOLOPCN(cls=args.cls)
    net_utils.load_net(trained_model, net)

    net.cuda()
    net.eval()

    #net.fix_net(net, net.state_dict())
    test_net(net, imdb, max_per_image, thresh, vis, args.image_size_index)
    imdb.close()
    
