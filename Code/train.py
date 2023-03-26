#coding:utf-8
import numpy as np
import torch
import GRCNN
import dataloader
from torch.autograd import Variable
import torch.optim as optim
import cv2
import os
from warpctc_pytorch import CTCLoss
from glob import glob
import utils
import shutil
from tqdm import tqdm
lr = 0.001
beta1 = 0.5
beta2 = 0.999
epoches = 200
is_use_gpu = False
gpu_list = -1
batch_size = 2
n_class = 37
n_hidden = 64 
img_w = 100
img_h = 32
max_len = 20
num_worker = 2  
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
model_save_path = './model_result/'
train_main_path = './IC03/'
val_main_path = './IC03/test/'


if is_use_gpu:
	os.environ['CUDA_VISIBLE_DEVICES']=gpu_list

if os.path.exists(model_save_path):
	shutil.rmtree(model_save_path)
os.mkdir(model_save_path)

dataset = dataloader.MyDataSet(train_main_path, width=img_w, height=img_h, max_len = max_len, label_file = './IC03/test_label.txt')
trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_img_list = glob(val_main_path + '*.jpg')

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)
crnn = GRCNN.GRCNN(n_class)

if is_use_gpu:
	crnn = crnn.cuda()
	criterion = CTCLoss().cuda()
else:
	criterion = CTCLoss()

print('net has load!')
converter = utils.strLabelConverter(alphabet)

optimizer=optim.Adam(crnn.parameters(), lr=lr, betas=(beta1, beta2))

best_acc=-1
totalLoss=[]
avg_test_acc = []
avg_train_acc = []

def get_img(img_path):
	img = cv2.imread(img_path)
	img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.resize(img, (img_w, img_h))
	img = np.reshape(img, newshape=[1,1, img_h, img_w])
	img = img.astype(np.float32)
	img = img / 255.0
	img = img - 0.5
	img = img * 2
	img_tensor = torch.from_numpy(img).float()
	label_file = './IC03/test_label.txt'
	file1 = open(label_file,"r")
	labels = file1.readlines()
	true_label = labels[int(os.path.basename(img_path).replace('.jpg', '').split('_')[-1])-1][:-1]
	return img_tensor, true_label


for epoch in range(1, epoches, 1):

	for p in crnn.parameters():
		p.requires_grad = True
	crnn.train()
	avg_totalLoss=0.0 
	train_acc = 0.0
	test_acc = 0.0
	for batch_id,(img_tensor, txt_len, txt_label, txt_name) in tqdm(enumerate(trainloader)):
		optimizer.zero_grad()
		batch_length = img_tensor.size(0)
		txt_label = txt_label.numpy().reshape(max_len*batch_length)
		txt_label = torch.from_numpy(np.array([item for item in txt_label if item != 0]).astype(np.int))
		if is_use_gpu:
			img_tensor = Variable(img_tensor.float()).cuda()
		else:
			img_tensor = Variable(img_tensor.float())
		txt_len = Variable(txt_len.int()).squeeze(1)
		txt_label = Variable(txt_label.int())

		preds = crnn(img_tensor)
		preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_length))
		total_loss = criterion(preds, txt_label, preds_size, txt_len) / batch_length
		total_loss.backward()
		optimizer.step()
		_, preds = preds.max(2)
		preds = preds.transpose(1, 0).contiguous().view(-1)
		sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
		for pred, target in zip(sim_preds, txt_name):
			if pred == target:
				train_acc += 1
		avg_totalLoss += total_loss.item()
		info='epoch : %d ,process: %d/%d ,  totalLoss: %f , lr: %f  ' % (epoch, batch_id, trainloader.__len__(), total_loss.item(), optimizer.param_groups[0]['lr'])

	train_acc /= len(dataset)
	avg_train_acc.append(train_acc)

	for p in crnn.parameters():
		p.requires_grad = False
	crnn.eval()
	for img_path in val_img_list:
		img_tensor, gt = get_img(img_path)
		if is_use_gpu:
			img_tensor = Variable(img_tensor).cuda()
		else:
			img_tensor = Variable(img_tensor)
		preds = crnn(img_tensor)
		_, preds = preds.max(2)
		preds = preds.transpose(1, 0).contiguous().view(-1)
		preds_size = Variable(torch.IntTensor([preds.size(0)]))
		raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
		sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

		print('%-33s => %-33s, gt: %-20s, %s' % (raw_pred, sim_pred, gt, sim_pred == gt))
		if sim_pred == gt:
			test_acc += 1


	test_acc /= len(val_img_list)
	avg_test_acc.append(test_acc)
	avg_totalLoss /= trainloader.__len__()
	totalLoss.append(avg_totalLoss)

	info='epoch : %d , avg_totalLoss: %f , lr: %f  avg_test_acc: %f   avg_train_acc: %f ' % (epoch, avg_totalLoss, optimizer.param_groups[0]['lr'], test_acc, train_acc)
	print(info)

	if best_acc <= test_acc:
		best_acc = test_acc
		if is_use_gpu:
			torch.save(crnn.cpu().state_dict(), model_save_path + 'cpu_model_parameter_'+ '.pkl')
			crnn.cuda()
		else:
			torch.save(crnn.state_dict(), model_save_path + 'cpu_model_parameter_' +  '.pkl')
		np.savetxt(model_save_path + 'total_loss.csv', totalLoss)
		np.savetxt(model_save_path + 'avg_test_acc.csv', avg_test_acc)
		np.savetxt(model_save_path + 'avg_train_acc.csv', avg_train_acc)

	if test_acc == 1:
		break
	if(epoch % 50 == 0):
		optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.6




