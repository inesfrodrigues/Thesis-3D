import torch
#from Thesis import hmsha_v2
from hmsha_v2 import HAT_Net_medium
from datasets import SEADataset
from timm.models import create_model
import utils
import matplotlib.pyplot as plt
import os

#def HAT_Net_medium(pretrained = False, **kwargs):
#    model = HAT_Net(dims = [64, 128, 320, 512], head = 64, kernel_sizes = [5, 3, 5, 3], expansions = [8, 8, 4, 4], grid_sizes = [8, 7, 7, 1], ds_ratios = [8, 4, 2, 1], depths = [3, 6, 18, 3],  **kwargs)
#    model.default_cfg = _cfg()
#    return model

model = create_model(
        'HAT_Net_medium',
        pretrained=False,
#        num_classes=args.nb_classes,
        drop_rate=0.0,
        #drop=args.drop,
        drop_path_rate=0.1,
#        drop_rate=args.drop,
#        drop_path_rate=args.drop_path,
#        drop_block_rate=None
    )

checkpoint = torch.load('checkpoints/HAT_Net_medium/checkpoint.pth')
#optimizer = TheOptimizerClass(*args, **kwargs)

model.load_state_dict(checkpoint['model'])
#optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']
#loss = checkpoint['loss']

model.eval()

dataset_val = SEADataset('/home/gpietrop/tensor/(12, 12, 20)/model2015/')

sampler_val = torch.utils.data.SequentialSampler(dataset_val)

data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * 2),
        num_workers=10,
        pin_memory=False,
        drop_last=False
    )

for images, target in data_loader_val:
	output = model(images)
	#path = "fig/"
	#path_directory = os.getcwd()
	dict_channel = {0: 'temperature', 1: 'salinity', 2: 'oxygen', 3: 'chla', 4: 'ppn'}
	directory = '/home/gpietrop/fig/output/checkpoint/'

	#if not os.path.exists(directory):
 	#   os.mkdir(directory)

	print('New output')
	number_fig = len(output[0, 0, :, 0, 0])  # number of levels of depth

	for i in range(number_fig):
	    cmap = plt.get_cmap('Greens')
	    output2 = output.detach().numpy()
	    plt.imshow(output2[0, 0, i, :, :], cmap=cmap)
	    #plt.title(dict_channel[3])
	    plt.title('chla')
	    plt.colorbar()
	    plt.savefig(directory + "/profondity_level_" + str(i) + ".png")
	    plt.close()
	    print('Plot done')
