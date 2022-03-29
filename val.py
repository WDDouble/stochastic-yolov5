# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python path/to/val.py --weights yolov5s.pt                 # PyTorch
                                      yolov5s.torchscript        # TorchScript
                                      yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s.xml                # OpenVINO
                                      yolov5s.engine             # TensorRT
                                      yolov5s.mlmodel            # CoreML (MacOS-only)
                                      yolov5s_saved_model        # TensorFlow SavedModel
                                      yolov5s.pb                 # TensorFlow GraphDef
                                      yolov5s.tflite             # TensorFlow Lite
                                      yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import json
import os
import sys
from pathlib import Path

import sys
sys.path.append('./cocoapi/PythonAPI/')

sys.path.append('./pdq_evaluation')
from read_files import convert_coco_det_to_rvc_det
import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from models.yolo import Model
from utils.callbacks import Callbacks
from utils.datasets import create_dataloader
from utils.general import (LOGGER, box_iou, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh, intersect_dicts,clip_coords,cov,is_pos_semidef,get_near_psd)
from utils.metrics import  ap_per_class
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.torch_utils import select_device, time_sync


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.from_numpy(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        num_samples=1,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        cfg=None,
        only_inference=False,
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        #model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        #stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine

        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = Model(cfg).to(device)  # create
        exclude = []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
        stride=max(int(model.stride.max()), 32)
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.half()

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train() #enable dropout

    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # warmup
        pad = 0.0 if task in ('speed', 'benchmark') else 0.5
        rect = False if task == 'benchmark' else True  # square inference for benchmarks
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, stride, single_cls, pad=pad, rect=rect,
                                       workers=workers, prefix=colorstr(f'{task}: '))[0]

    seen = 0

    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):

        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)




        t = time_sync()
            # Inference
        if num_samples == 1:
             inf_out, train_out = model(im) if training else model(im, augment=augment) # inference, loss outputs
        elif num_samples > 1:
             infs_all = []
             for i in range(num_samples):
                    out, _ = model(im, augment=augment)
                    infs_all.append(out.unsqueeze(2))
             inf_mean = torch.mean(torch.stack(infs_all), dim=0)
             infs_all.insert(0, inf_mean)
             inf_out = torch.cat(infs_all, dim=2)

        t0 += time_sync() - t

            # Loss
        if compute_loss:
              loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

            # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)  # to pixels

        t = time_sync()
        output, all_scores, sampled_coords = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres,
                                                                  multi_label=True,
                                                                  max_width=width, max_height=height)
        t1 += time_sync() - t

        for si, pred in enumerate(output):
             labels = targets[targets[:, 0] == si, 1:]
             nl = len(labels)
             tcls = labels[:, 0].tolist() if nl else []  # target class
             seen += 1

             if pred is None:
                  if nl:
                       stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                  continue



                # Clip boxes to image bounds
             clip_coords(pred, (height, width))

                # Append to pycocotools JSON dictionary
             if save_json:

                    # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                    image_id = int(Path(paths[si]).stem.split('_')[-1])
                    box = pred[:, :4].clone()  # xyxy
                    scale_coords(im[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                    box = xyxy2xywh(box)  # xywh
                    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner

                    # Getting covariances
                    # The transformations to coordinates follow the ones that are done below here after the if clause
                    if num_samples > 1:
                        # output: BS(list) x NUM_DETECTIONS x 6
                        # sampled_coords : BS(list) x NUM_DETECTIONS x NUM_SAMPLES x 4
                        # sampled_boxes : NUM_DETECTIONS x NUM_SAMPLES x 4
                        sampled_boxes = xywh2xyxy(sampled_coords[si].reshape(-1, 4)).reshape(sampled_coords[si].shape)
                        clip_coords(sampled_boxes.reshape(-1, 4), (height, width))

                        scale_coords(im[si].shape[1:], sampled_boxes.reshape(-1, 4), shapes[si][0], shapes[si][1])

                        # It will have 2 covariances matrices of 2X2 for each one of the two xy coordinates
                        covar_batch = torch.zeros(sampled_boxes.shape[0], 2, 2, 2)
                        for det_id in range(sampled_boxes.shape[0]):
                            covar_batch[det_id, 0, ...] = cov(sampled_boxes[det_id, :, :2])
                            covar_batch[det_id, 1, ...] = cov(sampled_boxes[det_id, :, 2:])

                        # Rounding it for smaller size
                        covar_batch = np.around(covar_batch.numpy(), 5).tolist()
                    else:
                        # Just dummy covars for the json zip() down below
                        covar_batch = [None] * pred.shape[0]

                    for p, b, p_all, covar_xyxy in zip(pred.tolist(), box.tolist(), all_scores[si].tolist(),
                                                       covar_batch):
                        if covar_xyxy is not None:
                            # Covariances need to be positive semi-definite, so just transform them here already
                            for i, covar_tmp in enumerate(covar_xyxy):
                                covar_tmp = np.array(covar_tmp)
                                if not is_pos_semidef(covar_tmp):
                                    print('Warning: Converted covar to near PSD')
                                    covar_xyxy[i] = get_near_psd(covar_tmp).tolist()

                        jdict.append({'image_id': image_id,
                                      'category_id': class_map[int(p[5])],
                                      'bbox': [round(x, 3) for x in b],
                                      'score': round(p[4], 5),
                                      'all_scores': [round(x, 5) for x in p_all],
                                      'covars': covar_xyxy})

             # Assign all predictions as incorrect
             correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
             if nl and not only_inference:
                 detected = []  # target indices
                 tcls_tensor = labels[:, 0]

                 # target boxes
                 tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                 # Per target class
                 for cls in torch.unique(tcls_tensor):
                     ti = (cls == tcls_tensor).nonzero().view(-1)  # target indices
                     pi = (cls == pred[:, 5]).nonzero().view(-1)  # prediction indices

                     # Search for detections
                     if pi.shape[0]:
                         # Prediction to target ious
                         ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                         # Append detections
                         for j in (ious > iouv[0]).nonzero():
                             d = ti[i[j]]  # detected target
                             if d not in detected:
                                 detected.append(d)
                                 correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                 if len(detected) == nl:  # all targets already located in image
                                     break

             # Append statistics (correct, conf, pcls, tcls)
             stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))


    if not only_inference:
            # Compute statistics
            stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
            if len(stats):
                p, r, ap, f1, ap_class = ap_per_class(*stats)
                if niou > 1:
                    p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
                mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
                nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
            else:
                nt = torch.zeros(1)

            # Print results
            pf = '%20s' + '%10.3g' * 6  # print format
            print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

            # Print results per class
            if verbose and nc > 1 and len(stats):
                for i, c in enumerate(ap_class):
                    print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

        # Print speeds
    if verbose or save_json:
            t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
            print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

        # Save JSON
    if save_json and len(jdict):
            pred_json = str(save_dir / f"dets_{name}_{conf_thres}_{iou_thres}.json")
            with open(pred_json, 'w') as file:
                json.dump(jdict, file)

            '''
            No need for this part as it will be evaluated later
            try:
                from pycocotools.coco import COCO
                from pycocotools.cocoeval import COCOeval
                # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                cocoGt = COCO(glob.glob(data['instances_path'])[0])  # initialize COCO ground truth api
                cocoDt = cocoGt.loadRes(f'output/dets_{name}_{conf_thres}_{iou_thres}.json')  # initialize COCO pred api
                cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
                cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                # mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)
            except Exception as e:
                print(e)
                print('WARNING: pycocotools must be installed with numpy==1.17 to run correctly. '
                      'See https://github.com/cocodataset/cocoapi/issues/356')
            '''
            del jdict
            print('Converting to RVC1 format...')
            convert_coco_det_to_rvc_det(det_filename=save_dir/f'dets_{name}_{conf_thres}_{iou_thres}.json',
                                       gt_filename=ROOT/'instances_val2017.json',
                                       save_filename=save_dir/f'dets_converted_{name}_{conf_thres}_{iou_thres}.json')


        # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
            maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s-custum.yaml', help='model.yaml path')
    parser.add_argument('--num_samples', type=int, default=1, help='How many times to sample if doing MC-Dropout')
    parser.add_argument('--only_inference', action='store_true')


    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(f'WARNING: confidence threshold {opt.conf_thres} >> 0.001 will produce invalid mAP values.')
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
                x, y = list(range(256, 1536 + 128, 128)), []  # x axis (image sizes), y axis
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
