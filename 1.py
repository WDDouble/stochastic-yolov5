import json
json_path='./dets_exp_0.1_0.6.json'
json_labels=json.load(open(json_path,"r"))
print(json_labels)