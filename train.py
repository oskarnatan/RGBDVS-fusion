#oskar@aisl.cs.tut.ac.jp

import os
import sys
import time
import yaml
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
import cv2
from torch import torch, optim, utils
torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0")
import arch
import utilx


trainmap = 'Tr2' #Tr1 Tr2 
valmap = 'Va1' #Va1 Va2

config = {
	'data_dir'			: ['dataset/'+trainmap[:-1]+'Set/', 'dataset/'+valmap[:-1]+'Set/'],  
	'data_info'			: ['dataset/'+trainmap+'/data_infon.yml', 'dataset/'+valmap+'/data_infon.yml'], 
	'input'				: ['dvs_f', 'dvs_l', 'dvs_ri', 'dvs_r', 'rgb_f', 'rgb_l', 'rgb_ri', 'rgb_r'],
	'task'				: ['depth_f', 'depth_l', 'depth_ri', 'depth_r', 'segmentation_f_min', 'segmentation_l_min', 'segmentation_ri_min', 'segmentation_r_min'],
	'mod_dir'			: 'model/',
	'arch'				: 'A0', #A0 or A1
	}
#load data info
with open(config['data_info'][0], 'r') as g:
	info = yaml.load(g, Loader=yaml.FullLoader)
with open(config['data_info'][1], 'r') as g:
	info_val = yaml.load(g, Loader=yaml.FullLoader)

#directory to save the model
config['mod_dir'] += config['arch']
os.makedirs(config['mod_dir'], exist_ok=True) 

def train(batches, model, lossf, metricf, optimizer):
	#training mode
	model.train()

	#variable to save the scores
	score = {'total_loss': utilx.AverageMeter(),
			'total_metric': utilx.AverageMeter(),
			'tot_DE_loss': utilx.AverageMeter(),
			'tot_DE_metric': utilx.AverageMeter(),
			'tot_SS_loss': utilx.AverageMeter(),
			'tot_SS_metric': utilx.AverageMeter()}

	#for visualization
	prog_bar = tqdm(total=len(batches))
	
	for batch_X, batch_Y_true, _ in batches:
		#move inputs and GT to the same device as the model
		for i in range(len(batch_X)):
			batch_X[i] = batch_X[i].double().to(device)
		for i in range(len(batch_Y_true)):
			batch_Y_true[i] = batch_Y_true[i].double().to(device)

		#forward pass
		batch_Y_pred = model(batch_X)

		#Loss and Metric calculation
		tot_DE_loss = lossf(batch_Y_pred[0], batch_Y_true[0])
		tot_DE_loss = torch.add(tot_DE_loss, lossf(batch_Y_pred[1], batch_Y_true[1]))
		tot_DE_loss = torch.add(tot_DE_loss, lossf(batch_Y_pred[2], batch_Y_true[2]))
		tot_DE_loss = torch.div(torch.add(tot_DE_loss, lossf(batch_Y_pred[3], batch_Y_true[3])), 4) #average across 4 views
		tot_DE_metric = metricf(batch_Y_pred[0], batch_Y_true[0])
		tot_DE_metric = torch.add(tot_DE_metric, metricf(batch_Y_pred[1], batch_Y_true[1]))
		tot_DE_metric = torch.add(tot_DE_metric, metricf(batch_Y_pred[2], batch_Y_true[2]))
		tot_DE_metric = torch.div(torch.add(tot_DE_metric, metricf(batch_Y_pred[3], batch_Y_true[3])), 4) #average across 4 views
		tot_SS_loss = lossf(batch_Y_pred[4], batch_Y_true[4])
		tot_SS_loss = torch.add(tot_SS_loss, lossf(batch_Y_pred[5], batch_Y_true[5]))
		tot_SS_loss = torch.add(tot_SS_loss, lossf(batch_Y_pred[6], batch_Y_true[6]))
		tot_SS_loss = torch.div(torch.add(tot_SS_loss, lossf(batch_Y_pred[7], batch_Y_true[7])), 4) #dirata-rata dari 4 view
		tot_SS_metric = metricf(batch_Y_pred[4], batch_Y_true[4])
		tot_SS_metric = torch.add(tot_SS_metric, metricf(batch_Y_pred[5], batch_Y_true[5]))
		tot_SS_metric = torch.add(tot_SS_metric, metricf(batch_Y_pred[6], batch_Y_true[6]))
		tot_SS_metric = torch.div(torch.add(tot_SS_metric, metricf(batch_Y_pred[7], batch_Y_true[7])), 4) #dirata-rata dari 4 view
		total_loss = torch.add(tot_DE_loss, tot_SS_loss)
		total_metric = torch.add(tot_DE_metric, torch.sub(1,tot_SS_metric)) 

		#update weights
		optimizer.zero_grad()
		total_loss.backward()
		optimizer.step() 

		#to be saved
		score['total_loss'].update(total_loss.item(), 1) 
		score['total_metric'].update(total_metric.item(), 1) 
		score['tot_DE_loss'].update(tot_DE_loss.item(), 1) 
		score['tot_DE_metric'].update(tot_DE_metric.item(), 1) 
		score['tot_SS_loss'].update(tot_SS_loss.item(), 1) 
		score['tot_SS_metric'].update(tot_SS_metric.item(), 1)

		postfix = OrderedDict([('t_total_l', score['total_loss'].avg),
							('t_total_m', score['total_metric'].avg),
							('t_DE_l', score['tot_DE_loss'].avg),
							('t_DE_m', score['tot_DE_metric'].avg),
							('t_SS_l', score['tot_SS_loss'].avg),
							('t_SS_m', score['tot_SS_metric'].avg)])

		prog_bar.set_postfix(postfix)
		prog_bar.update(1)

	prog_bar.close()

	return postfix

def validate(batches, model, lossf, metricf):
	#validation mode
	model.eval()

	#variable to save the scores
	score = {'total_loss': utilx.AverageMeter(),
			'total_metric': utilx.AverageMeter(),
			'tot_DE_loss': utilx.AverageMeter(),
			'tot_DE_metric': utilx.AverageMeter(),
			'tot_SS_loss': utilx.AverageMeter(),
			'tot_SS_metric': utilx.AverageMeter()}

	#for visualization
	prog_bar = tqdm(total=len(batches))
	
	with torch.no_grad():
		for batch_X, batch_Y_true, _ in batches:
			#move inputs and GT to the same device as the model
			for i in range(len(batch_X)):
				batch_X[i] = batch_X[i].double().to(device)
			for i in range(len(batch_Y_true)):
				batch_Y_true[i] = batch_Y_true[i].double().to(device)

			#forward pass
			batch_Y_pred = model(batch_X)

			#Loss and Metric calculation
			tot_DE_loss = lossf[0](batch_Y_pred[0], batch_Y_true[0])
			tot_DE_loss = torch.add(tot_DE_loss, lossf[0](batch_Y_pred[1], batch_Y_true[1]))
			tot_DE_loss = torch.add(tot_DE_loss, lossf[0](batch_Y_pred[2], batch_Y_true[2]))
			tot_DE_loss = torch.div(torch.add(tot_DE_loss, lossf[0](batch_Y_pred[3], batch_Y_true[3])), 4) #average across 4 views
			tot_DE_metric = metricf[0](batch_Y_pred[0], batch_Y_true[0])
			tot_DE_metric = torch.add(tot_DE_metric, metricf[0](batch_Y_pred[1], batch_Y_true[1]))
			tot_DE_metric = torch.add(tot_DE_metric, metricf[0](batch_Y_pred[2], batch_Y_true[2]))
			tot_DE_metric = torch.div(torch.add(tot_DE_metric, metricf[0](batch_Y_pred[3], batch_Y_true[3])), 4) #average across 4 views
			tot_SS_loss = lossf[1](batch_Y_pred[4], batch_Y_true[4])
			tot_SS_loss = torch.add(tot_SS_loss, lossf[1](batch_Y_pred[5], batch_Y_true[5]))
			tot_SS_loss = torch.add(tot_SS_loss, lossf[1](batch_Y_pred[6], batch_Y_true[6]))
			tot_SS_loss = torch.div(torch.add(tot_SS_loss, lossf[1](batch_Y_pred[7], batch_Y_true[7])), 4) #dirata-rata dari 4 view
			tot_SS_metric = metricf[1](batch_Y_pred[4], batch_Y_true[4])
			tot_SS_metric = torch.add(tot_SS_metric, metricf[1](batch_Y_pred[5], batch_Y_true[5]))
			tot_SS_metric = torch.add(tot_SS_metric, metricf[1](batch_Y_pred[6], batch_Y_true[6]))
			tot_SS_metric = torch.div(torch.add(tot_SS_metric, metricf[1](batch_Y_pred[7], batch_Y_true[7])), 4) #dirata-rata dari 4 view
			total_loss = torch.add(tot_DE_loss, tot_SS_loss)
			total_metric = torch.add(tot_DE_metric, torch.sub(1,tot_SS_metric)) 

			#to be saved
			score['total_loss'].update(total_loss.item(), 1) 
			score['total_metric'].update(total_metric.item(), 1) 
			score['tot_DE_loss'].update(tot_DE_loss.item(), 1) 
			score['tot_DE_metric'].update(tot_DE_metric.item(), 1) 
			score['tot_SS_loss'].update(tot_SS_loss.item(), 1) 
			score['tot_SS_metric'].update(tot_SS_metric.item(), 1)

			postfix = OrderedDict([('v_total_l', score['total_loss'].avg),
								('v_total_m', score['total_metric'].avg),
								('v_DE_l', score['tot_DE_loss'].avg),
								('v_DE_m', score['tot_DE_metric'].avg),
								('v_SS_l', score['tot_SS_loss'].avg),
								('v_SS_m', score['tot_SS_metric'].avg)])

			prog_bar.set_postfix(postfix)
			prog_bar.update(1)

		prog_bar.close()

	return postfix

#MAIN FUNCTION
def main():
	#IMPORT MODEL ARCHITECTURE
	if config['arch'] == 'A0': 
		model = arch0.A0()
	elif config['arch'] == 'A1': 
		model = arch0.A1() 
	else:
		sys.exit("ARCH NOT FOUND............................")
	model.double().to(device) #load model ke CUDA chace memory

	#loss and metric function
	lossf = [utilx.HuberLoss().to(device), utilx.BCEDiceLoss().to(device)]
	metricf = [utilx.L1Loss().to(device), utilx.IOUScore().to(device)]

	#optimizer and learning rate scheduler
	params = filter(lambda p: p.requires_grad, model.parameters())
	optima = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0001)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optima, mode='min', factor=0.5, patience=5, min_lr=0.00001)

	#create batch of train and val data
	train_dataset = utilx.datagen(file_ids=info['train_idx'], config=config, data_info=info, input_dir=config['data_dir'][0])
	train_batches = utils.data.DataLoader(train_dataset,
		batch_size=config['tensor_dim'][0], 
		shuffle=True,
		num_workers=4,
		drop_last=False)
	val_dataset = utilx.datagen(file_ids=info_val['val_idx'], config=config, data_info=info_val, input_dir=config['data_dir'][1])
	val_batches = utils.data.DataLoader(val_dataset,
		batch_size=config['tensor_dim'][0],
		shuffle=False,
		num_workers=4,
		drop_last=False)

	#create training log
	log = OrderedDict([
		('epoch', []),
		('lrate', []),
		('train_total_loss', []), 
		('val_total_loss', []),
		('train_total_metric', []), 
		('val_total_metric', []),
		('train_depth_loss', []),
		('val_depth_loss', []),
		('train_depth_metric', []),
		('val_depth_metric', []),
		('train_seg_loss', []),
		('val_seg_loss', []),
		('train_seg_metric', []),
		('val_seg_metric', []),
		('best_model', []),
		('stop_counter', []),
		('elapsed_time', []),
	])
	
	#LOOP
	lowest_monitored_score = float('inf')
	stop_count = 35
	while True:
		print('\n=======---=======---=======Epoch:%.4d=======---=======---=======' % (epoch))
		print("current lr: ", optima.param_groups[0]['lr'])

		#train - val
		start_time = time.time() 
		train_log = train(batches=train_batches, model=model, lossf=lossf, metricf=metricf, optimizer=optima)
		val_log = validate(batches=val_batches, model=model, lossf=lossf, metricf=metricf)
		elapsed_time = time.time() - start_time 

		#update learning rate
		scheduler.step(val_log['v_total_m']) #parameter acuan reduce LR adalah val_total_metric

		log['epoch'].append(epoch+1)
		log['lrate'].append(current_lr)
		log['train_total_loss'].append(train_log['t_total_l'])
		log['val_total_loss'].append(val_log['v_total_l'])
		log['train_total_metric'].append(train_log['t_total_m'])
		log['val_total_metric'].append(val_log['v_total_m'])
		log['train_depth_loss'].append(train_log['t_DE_l'])
		log['val_depth_loss'].append(val_log['v_DE_l'])
		log['train_depth_metric'].append(train_log['t_DE_m'])
		log['val_depth_metric'].append(val_log['v_DE_m'])
		log['train_seg_loss'].append(train_log['t_SS_l'])
		log['val_seg_loss'].append(val_log['v_SS_l'])
		log['train_seg_metric'].append(train_log['t_SS_m'])
		log['val_seg_metric'].append(val_log['v_SS_m'])
		log['elapsed_time'].append(elapsed_time)

		print('| t_total_l: %.4f | t_total_m: %.4f | t_DE_l: %.4f | t_DE_m: %.4f | t_SS_l: %.4f | t_SS_m: %.4f |'
			% (train_log['t_total_l'], train_log['t_total_m'], train_log['t_DE_l'], train_log['t_DE_m'], train_log['t_SS_l'], train_log['t_SS_m']))
		print('| v_total_l: %.4f | v_total_m: %.4f | v_DE_l: %.4f | v_DE_m: %.4f | v_SS_l: %.4f | v_SS_m: %.4f |'
			% (val_log['v_total_l'], val_log['v_total_m'], val_log['v_DE_l'], val_log['v_DE_m'], val_log['v_SS_l'], val_log['v_SS_m']))
		print('elapsed time: %.4f sec' % (elapsed_time))
		
		
		#save the best model only
		if val_log['v_total_m'] < lowest_monitored_score:
			print("v_total_m: %.4f < previous lowest: %.4f" % (val_log['v_total_m'], lowest_monitored_score))
			print("model saved!")
			torch.save(model.state_dict(), config['mod_dir']+'/model_weights.pth')
			lowest_monitored_score = val_log['v_total_m']
			#reset stop counter
			stop_count = 35
			print("stop counter reset: ", stop_count)
			log['best_model'].append("BEST")
		else:
			stop_count -= 1
			print("v_total_m: %.4f >= previous lowest: %.4f, training stop in %d epoch" % (val_log['v_total_m'], lowest_monitored_score, stop_count))
			log['best_model'].append("")

		#update stop counter
		log['stop_counter'].append(stop_count)
		#paste to csv file
		pd.DataFrame(log).to_csv(config['mod_dir']+'/model_log.csv', index=False)

		if stop_count==0:
			print("NO IMPROVEMENT!, TRAINING STOP!")
			break

		torch.cuda.empty_cache()


if __name__ == "__main__":
	main()












""""""
