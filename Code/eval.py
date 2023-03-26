import numpy as np
import torch
import GRCNN
from torch.autograd import Variable
import cv2
import os
from glob import glob
import utils
import shutil
import argparse



gpu_list = -1
batch_size = 2
n_class = 37
n_hidden = 64 
img_w = 100
img_h = 32
max_len = 20
num_worker = 2  
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
pre_train_model_path = './model_result/cpu_model_parameter_.pkl'
test_data_path = './IC03/test'
test_result_save_path = './model_result/'


if is_use_gpu:
	os.environ['CUDA_VISIBLE_DEVICES']=gpu_list

if os.path.exists(test_result_save_path):
	shutil.rmtree(test_result_save_path)
os.mkdir(test_result_save_path)

test_img_list = glob(test_data_path + '*.jpg')


crnn = GRCNN.GRCNN(nclass=n_class)
crnn.load_state_dict(torch.load(pre_train_model_path))

if is_use_gpu:
	crnn = crnn.cuda()
print('net has load!')
converter = utils.strLabelConverter(alphabet)
crnn.eval()


def get_img(img_path):
	img = cv2.imread(img_path)
	img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	img = cv2.resize(img, (img_w, img_h))
	img = np.reshape(img, newshape=[1,1, img_h, img_w])
	img = img.astype(np.float32)
	img = img / 255
	img = img - 0.5
	img = img * 2
	img_tensor = torch.from_numpy(img).float()
	return img_tensor


for img_path in test_img_list:
	img_tensor = get_img(img_path)
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
	print('%-33s => %-33s' % (raw_pred, sim_pred))




