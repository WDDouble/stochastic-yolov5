import subprocess

class model:
    def __init__(self, drop_rate):
        self.drop_rate=str(drop_rate)

    def run(self):
        subprocess.Popen(['python', 'val.py','--cfg','yolov5s-custum.yaml','--batch','16','--data','coco.yaml','--imgsz','640','--iou-thres','0.6','--num_samples','10','--conf-thres','0.5','--new_drop_rate',self.drop_rate])

Model=model(0.1)
Model.run()