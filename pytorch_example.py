import torch
import torch.nn as nn
import tensornetwork as tn
from torchvision import datasets, transforms
import torch.utils.data as Data
from torch import optim
from tqdm import tqdm
import TNModel.pytorch_model as pytorch_model

tn.set_default_backend('pytorch')

# HyperParams
hyper_params = {
	'rank': 28*28,
	'phys_dim': 2,
	'bond_dim': 6,
	'string_cnt': 2, # of strings in SBS 
	'labels': 10,
	'device': 'cpu',
	'batch_size': 16,
	'model': 'mps'
}

if hyper_params['model']!='peps':
	transform = transforms.Compose([
		transforms.ToTensor()
	])
else:
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.5,), std=(1.0,))
	])

mnist_train = datasets.MNIST('./data/', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('./data/', train=False, download=True, transform=transform)
mnist_test = Data.Subset(dataset=mnist_test, indices=[i for i in range(300)])
train_loader = Data.DataLoader(dataset=mnist_train, batch_size=hyper_params['batch_size'], shuffle=True)
test_loader = Data.DataLoader(dataset=mnist_test, batch_size=hyper_params['batch_size'], shuffle=False)

# Build Model
print('Building Model...')

net = None
if hyper_params['model']=='mps':
	net = pytorch_model.MPSLayer(hyper_params)
elif hyper_params['model']=='sbs1d':
	net = pytorch_model.SBS1dLayer(hyper_params)
elif hyper_params['model']=='peps':
	net = pytorch_model.PEPSCNNLayer(hyper_params)

optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=0.0)
loss_func = nn.CrossEntropyLoss()


def evaluate():
	print('Evaluating...')
	net.eval()
	total_item = 0
	total_acc = 0
	
	with torch.no_grad():
		for batchx, batchy in tqdm(test_loader):
			if hyper_params['model']!='peps':
				batchx = batchx.view(-1, 28*28)
				xcos = 1.0 - batchx
				xsin = batchx

				batchx = torch.stack([xcos, xsin], dim=-1)

			pred = net(batchx)
			total_item += batchx.shape[0]
			total_acc += torch.sum(torch.argmax(pred, dim=1)==batchy).item()

		print(f'Acc: {total_acc}/{total_item}   {total_acc/total_item}')
	net.train()


print('Start training...')
for epoch in range(10):
	print(f'Epoch {epoch}')
	for step, (batchx, batchy) in enumerate(train_loader):
		if not hyper_params['model']=='peps':
			batchx = batchx.view(-1, 28*28)
			xcos = 1.0 - batchx
			xsin = batchx

			batchx = torch.stack([xcos, xsin], dim=-1)

		pred = net(batchx)
		loss = loss_func(pred, batchy)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		print(f'Step:{step} Loss:{loss.item()}')
		if step % 50 == 0:
			evaluate()
