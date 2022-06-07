#oskar@aisl.cs.tut.ac.jp

import os
import yaml
import cv2
from torch import torch, utils
import numpy as np
import pandas as pd
from collections import OrderedDict
import time
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0")

import arch, utilx


main_ddir = "/home/aisl/tempo/RGBDVS-fusion/dataset/"
trainmap = "Tr1" #Tr1 Tr2
valmap = "Va2" #Va1 Va2
mapx = "Te2/"  #Te1 Te2
mod_folder = "A1/" #A1 A0
mod_dir = "model/perception_"+trainmap+valmap+"/"+mod_folder #model untuk inference
save_dir = "prediction/"+mapx #masukkan ke folder map
with open(main_ddir+mapx+"data_infon.yml", 'r') as f:
	test_info = yaml.load(f)
listfiles = test_info['test_idx']#val_idx train_idx test_idx 
listfiles.sort() #urutkan



#PRINT CONFIGURATION
print("==========================================")
print("MODEL CONFIGURATION:")
with open(mod_dir+"model_config.yml", 'r') as f:
	config = yaml.load(f)
for key in config.keys():
	print('%s: %s' % (key, str(config[key])))
with open(mod_dir+"data_info.yml", 'r') as f:
	data_info = yaml.load(f)

#BUAT SAVE DIRECTORY
save_dir = save_dir+trainmap+valmap+"_models/"+mod_folder
os.makedirs(save_dir, exist_ok=True) 



# LOAD ARSITEKTUR DAN WEIGHTS MODEL
if config['arch'] == 'A0': 
	model = arch.A0() 
elif config['arch'] == 'A1':
	model = arch.A1()
model.double().to(device) #load model ke CUDA chace memory
model.load_state_dict(torch.load(mod_dir+"model_weights.pth", map_location=device))
model.eval()




inputd = utilx.datagen( #pakai datagen4 jika pakai arch4
	file_ids=listfiles,
	input_dir = main_ddir+mapx[:-2]+"Set/",
	config=config,
	data_info=data_info)
data = utils.data.DataLoader(inputd,
	batch_size=1, 
	shuffle=False, 
	num_workers=4,
	drop_last=False)


lossf = [utilx.HuberLoss().to(device), utilx.BCEDiceLoss().to(device)]
metricf = [utilx.L1Loss().to(device), utilx.IOUScore().to(device)]
batch = 0
log = OrderedDict([
	('batch', []),
	('forwardpass_time', []),
	('test_total_metric', []),
	('test_depth_metric', []),
	('test_seg_metric', []),
	('test_total_loss', []),
	('test_depth_loss', []),
	('test_seg_loss', [])])


#AMBIL INPUT X AJA UNTUK INFERENCE
for input_x, batch_Y_true, img_id in data:
	file_name = img_id['img_id']
	for i in range(len(input_x)): #len(batch_Y_true)
		input_x[i] = input_x[i].to(device)#cuda()
		batch_Y_true[i] = batch_Y_true[i].to(device)#cuda()

	start_time = time.time()
	batch_Y_pred = model(input_x)
	infer_time = time.time() - start_time
	
			
	#DEPTH
	tot_DE_loss = lossf[0](batch_Y_pred[0], batch_Y_true[0])
	tot_DE_loss = torch.add(tot_DE_loss, lossf[0](batch_Y_pred[1], batch_Y_true[1]))
	tot_DE_loss = torch.add(tot_DE_loss, lossf[0](batch_Y_pred[2], batch_Y_true[2]))
	tot_DE_loss = torch.div(torch.add(tot_DE_loss, lossf[0](batch_Y_pred[3], batch_Y_true[3])), 4) 
	tot_DE_metric = metricf[0](batch_Y_pred[0], batch_Y_true[0])
	tot_DE_metric = torch.add(tot_DE_metric, (metricf[0](batch_Y_pred[1], batch_Y_true[1])))
	tot_DE_metric = torch.add(tot_DE_metric, (metricf[0](batch_Y_pred[2], batch_Y_true[2])))
	tot_DE_metric = torch.div(torch.add(tot_DE_metric, (metricf[0](batch_Y_pred[3], batch_Y_true[3]))), 4) 
	#SEG
	tot_SS_loss = lossf[1](batch_Y_pred[4], batch_Y_true[4])
	tot_SS_loss = torch.add(tot_SS_loss, lossf[1](batch_Y_pred[5], batch_Y_true[5]))
	tot_SS_loss = torch.add(tot_SS_loss, lossf[1](batch_Y_pred[6], batch_Y_true[6]))
	tot_SS_loss = torch.div(torch.add(tot_SS_loss, lossf[1](batch_Y_pred[7], batch_Y_true[7])), 4) 
	tot_SS_metric = metricf[1](batch_Y_pred[4], batch_Y_true[4])
	tot_SS_metric = torch.add(tot_SS_metric, (metricf[1](batch_Y_pred[5], batch_Y_true[5])))
	tot_SS_metric = torch.add(tot_SS_metric, (metricf[1](batch_Y_pred[6], batch_Y_true[6])))
	tot_SS_metric = torch.div(torch.add(tot_SS_metric, (metricf[1](batch_Y_pred[7], batch_Y_true[7]))), 4)
	total_loss = torch.add(tot_DE_loss, tot_SS_loss)
	total_metric = torch.add(tot_DE_metric, torch.sub(1,tot_SS_metric)) 

	#simpan pada log csv
	log['batch'].append(batch)
	log['forwardpass_time'].append(infer_time)
	log['test_total_loss'].append(total_loss.item())
	log['test_total_metric'].append(total_metric.item())
	log['test_depth_loss'].append(tot_DE_loss.item())
	log['test_depth_metric'].append(tot_DE_metric.item())
	log['test_seg_loss'].append(tot_SS_loss.item())
	log['test_seg_metric'].append(tot_SS_metric.item())
	batch += 1
	
	#paste ke csv file
	pd.DataFrame(log).to_csv(save_dir+'/test_performance.csv', index=False)

	#detach tensor
	for i in range(len(config['task'])):
		batch_Y_pred[i] = batch_Y_pred[i].cpu().detach().numpy()

	#loop batch
	for batch_i in range(1):
		print(file_name[batch_i])
		#LOOP TASK
		for task_i in range(len(config['task'])):
			task_folder = save_dir+"pred_"+config['task'][task_i]+"/"
			os.makedirs(task_folder, exist_ok=True)
			if config['task'][task_i][:3] == 'dep': #depth estimation
				pred_dep = batch_Y_pred[task_i][batch_i].transpose(1,2,0)
				norm_dep = pred_dep * 255.0 
				cv2.imwrite(task_folder+file_name[batch_i], norm_dep) #cetak predicted dep
			
			elif config['task'][task_i][:3] == 'seg': #segmentation
				imgx = np.zeros((config['tensor_dim'][2], config['tensor_dim'][3], 3))
				pred_seg = batch_Y_pred[task_i][batch_i]
				col_mask = []
				for i in range(data_info['n_seg_class']):
					img_blank = utilx.blank_frame(rgb_color=data_info['seg_colors'][i], target_dim=[config['tensor_dim'][2], config['tensor_dim'][3]])
					col_mask.append(img_blank)
				col_mask = np.array(col_mask)
				#looping class
				for cls_i in range(pred_seg.shape[0]):
					pred_mask = pred_seg[cls_i]
					pred_mask = np.expand_dims(pred_mask, axis=-1) 
					col_mask[cls_i][:,:,0:1] = col_mask[cls_i][:,:,0:1] * pred_mask 
					col_mask[cls_i][:,:,1:2] = col_mask[cls_i][:,:,1:2] * pred_mask 
					col_mask[cls_i][:,:,2:3] = col_mask[cls_i][:,:,2:3] * pred_mask 
					imgx = imgx+col_mask[cls_i]
				imgx = utilx.swap_RGB2BGR(imgx)
				cv2.imwrite(task_folder+file_name[batch_i], imgx)


log['batch'].append("avg_total")
log['forwardpass_time'].append(np.mean(log['forwardpass_time']))
log['test_total_loss'].append(np.mean(log['test_total_loss']))
log['test_total_metric'].append(np.mean(log['test_total_metric']))
log['test_depth_loss'].append(np.mean(log['test_depth_loss']))
log['test_depth_metric'].append(np.mean(log['test_depth_metric']))
log['test_seg_loss'].append(np.mean(log['test_seg_loss']))
log['test_seg_metric'].append(np.mean(log['test_seg_metric']))

log['batch'].append("var_total")
log['forwardpass_time'].append(np.var(log['forwardpass_time'][:-1]))
log['test_total_loss'].append(np.var(log['test_total_loss'][:-1]))
log['test_total_metric'].append(np.var(log['test_total_metric'][:-1]))
log['test_depth_loss'].append(np.var(log['test_depth_loss'][:-1]))
log['test_depth_metric'].append(np.var(log['test_depth_metric'][:-1]))
log['test_seg_loss'].append(np.var(log['test_seg_loss'][:-1]))
log['test_seg_metric'].append(np.var(log['test_seg_metric'][:-1]))

pd.DataFrame(log).to_csv(save_dir+'/test_performance.csv', index=False)



