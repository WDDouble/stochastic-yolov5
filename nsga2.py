import subprocess
import numpy as np
from pymoo.core.problem import Problem
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter


class MyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2,xl=np.array([0, 0, 1]), xu=[1,2,10])

    def _evaluate(self, x, out, *args, **kwargs):
        Model=model(x)
        output=Model.run()
        out["F"] = np.column_stack([output[0], output[1]])

mask = ["real", "int","int"]
sampling = MixedVariableSampling(mask, {
    "real": get_sampling("real_random"),
    "int": get_sampling("int_random")
})

crossover = MixedVariableCrossover(mask, {
    "real": get_crossover("real_sbx", prob=1.0, eta=3.0),
    "int": get_crossover("int_sbx", prob=1.0, eta=3.0)
})

mutation = MixedVariableMutation(mask, {
    "real": get_mutation("real_pm", eta=3.0),
    "int": get_mutation("int_pm", eta=3.0)
})
problem = MyProblem()

algorithm = NSGA2(pop_size=20)

res = minimize(problem,
               algorithm,
               ('n_gen', 10),
               seed=1,
               verbose=False))

plot = Scatter()
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()


class model:
    def __init__(self, drop_rate:float,dropout_type:int,num_sample:int):
        self.drop_rate=str(drop_rate)
        self.dropout_type=dropout_type
        self.num_sample=str(num_sample)

    def run(self):
        cfg_list=["yolov5s-dropout.yaml","yolov5s-gdropout.yaml","yolov5s-dropblock.yaml"]
        print("running yolov5...")
        subprocess.call(['python', 'val.py','--cfg',cfg_list[self.dropout_type],'--batch','16','--data','coco.yaml','--imgsz','640','--iou-thres','0.6','--num_samples',self.num_sample,'--conf-thres','0.5','--new_drop_rate',self.drop_rate],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        print("evaluating PDQ and mAP...")
        subprocess.call(['python', 'pdq_evaluation/evaluate.py','--test_set','coco','--gt_loc','/content/datasets/coco/annotations/instances_val2017.json','--det_loc','/content/stochastic-yolov5/dets_converted_exp_0.5_0.6.json','--save_folder','output','--num_workers','15'],stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT)
        data={}
        with open(r"./output/scores.txt") as f:
            for line in f.readlines():
                if line:
                    name,value=line.strip().split(':',1)
                    data[name]=float(value)
        return [data['PDQ'],data['mAP']]


