import os
import torch 
import torchvision

import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from architectures import *
import argparse, sys
from utils import *
import os


def main(args):
	Epochs = 100
	Lr_Rate = 1e-3
	Batch_Size = 128

	transform = transforms.Compose([
		transforms.ToTensor(),
		#transforms.Normalize((0.5,), (0.5,))
	])

	def get_device():
		if torch.cuda.is_available():
			device = 'cuda:0'
		else:
			device = 'cpu'
		return device

	device = get_device()
	TRAIN_PATH = args.train_dataset
	VAL_PATH = args.val_dataset



	train_set = datasets.ImageFolder(root=TRAIN_PATH, transform=transform)
	test_set = datasets.ImageFolder(root=VAL_PATH, transform=transform)

	train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
	test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

	
	model = SimpleAutoEncoder()
	print(model)

	criterion = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
															mode='min',
															factor=0.2,
															patience=20,
															min_lr=5e-5)



	model.to(device)
	make_dir()
	
	train_loss = training(model, train_loader, args.epochs, optimizer)


	torch.save(model, args.path_save)
	
	plt.figure()
	plt.plot(train_loss)
	plt.title('Train Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.savefig('deep_ae_mnist_loss.png')



if __name__ == "__main__":
	parser=argparse.ArgumentParser()
	BATCH_SIZE = 64
	parser.add_argument('--lr', help='Learning Rate, default = 0.0001', type=float, default=0.001)
	parser.add_argument('--epochs', help='Number of epochs, default = 100', type=int, default=300)
	parser.add_argument('--load_trained', help='Load existing model', type=bool, default=False)
	parser.add_argument('--train_dataset', help='Path to the train dataset (ImageFolder scheme).', type=str)
	parser.add_argument('--val_dataset', help='Path to the validation dataset (ImageFolder scheme).', type=str)
	parser.add_argument('--path_save', help='Where to save model.', type=str, default="autoencoder.model")
	#../../DATASETS/CLASSIF_BIGGEST_SQUARE/"
	args=parser.parse_args()
	main(args)