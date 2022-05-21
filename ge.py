import subprocess

class model:
    def __init__(self, drop_rate):
        self.drop_rate=str(drop_rate)

    def run(self):
        subprocess.Popen(['python', 'val.py','--cfg','yolov5s-custum.yaml','--batch','16','--data','coco.yaml','--imgsz','640','--iou-thres','0.6','--num_samples','10','--conf-thres','0.5','--new_drop_rate',self.drop_rate])
        subprocess.Popen(['python', 'pdq_evaluation/evaluate.py','--test_set','coco','--gt_loc','/content/datasets/coco/annotations/instances_val2017.json','--det_loc','/content/stochastic-yolov5/dets_converted_exp_0.5_0.6.json','--save_folder','output','--num_workers','15'])

Model=model(0.1)
Model.run()