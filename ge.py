import subprocess

class model:
    def __init__(self, drop_rate:float,dropout_type:int):
        self.drop_rate=str(drop_rate)
        self.dropout_type=dropout_type

    def run(self):
        cfg_list=["yolov5s_dropout","yolov5s_gdropout","yolov5s_dropblock"]

        subprocess.call(['python', 'val.py','--cfg',cfg_list[self.dropout_type],'--batch','16','--data','coco.yaml','--imgsz','640','--iou-thres','0.6','--num_samples','10','--conf-thres','0.5','--new_drop_rate',self.drop_rate],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        subprocess.call(['python', 'pdq_evaluation/evaluate.py','--test_set','coco','--gt_loc','/content/datasets/coco/annotations/instances_val2017.json','--det_loc','/content/stochastic-yolov5/dets_converted_exp_0.5_0.6.json','--save_folder','output','--num_workers','15'],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        data={}
        with open(r"./output/scores.txt") as f:
            for line in f.readlines():
                if line:
                    name,value=line.strip().split(':',1)
                    data[name]=float(value)
        return [data['PDQ'],data['mAP']]


