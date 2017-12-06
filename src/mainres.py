import torch.utils.data
import torchvision.datasets as datasets

from ResCNN import *
from utils import *


# CUDA Configurations
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

num_epochs = 20
N = 4
kwargs = {'num_workers': 1} if torch.cuda.is_available() else {}

if args.mode=="train":
	train=1
	test=0
elif args.mode=="test":
	train=0
	test=1

def main(kwargs):
	
	if train==1:
		res_cnn = ResCNN(args.norm)		

		# Contents
		coco = datasets.ImageFolder(root='images/coco10k/', transform=loader)
		# faces = datasets.ImageFolder(root='images/faces/', transform=loader)
		content_loader = torch.utils.data.DataLoader(coco, batch_size=N, shuffle=True,drop_last=True,pin_memory=False, **kwargs)

		for epoch in range(num_epochs):
			for i, content_batch in enumerate(content_loader):
				# print(content_batch)				
				iteration = epoch * i + i
				# content_batch[0]=content_batch[0].type(dtype)
				
				content_loss,low = res_cnn.train(content_batch[0])

				if i % 10 == 0:
				  print("Iteration: %d" % (iteration))
				  print("Content loss: %f" % (content_loss.data[0]))
				  # print("Style loss: %f" % (style_loss.data[0]))

				if i % 2000 == 0:
				  path = "outputsr/%d_" % (iteration)
				  paths = [path + str(n) + ".png" for n in range(N)]
				  # print(pastiches.size())				
				  save_images(low, paths)

				  # path = "outputsr/content_%d_" % (iteration)
				  # paths = [path + str(n) + ".png" for n in range(N)]
				  # save_images(high, paths) ##TODO save content images
				  

				  modelname="models/it%d.pt" % (iteration)
				  torch.save(res_cnn, modelname)

	if test==1:
		res_cnn = torch.load(args.model)
		content = image_loader("testImages/content10.jpg").type(dtype)
		path = "testImages/content10.jpg"
		# save_image(content, path)
		pastiche=res_cnn.test(content)
		path = "testImages/trained.png"
		save_image_out(pastiche, path)






main(kwargs)