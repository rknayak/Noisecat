import torch
import numpy as np
from cats_dataloader import CatsWithNoiseDataset
from model_denoise import Net
import sys
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

def evaluate_on_dataset(dataset_name):


	test_ds = CatsWithNoiseDataset(dataset_name)
	dataloader = DataLoader(test_ds,batch_size=20)


	net = Net()

	net.load_state_dict(torch.load('trained_model.pt',map_location=torch.device('cpu')))

	net.eval()

	loss_func = nn.MSELoss()

	loss = 0
	n_batches = 0
	with torch.no_grad():
		for x,y in dataloader:
			
			n_batches+=1
			
			pred = net(x)
			
			loss+= loss_func(pred,y).item()

	return loss/n_batches


if __name__ == "__main__":

	loss = evaluate_on_dataset()

	print(loss)

