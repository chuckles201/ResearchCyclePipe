import torch
import PIL
from dataloader import ImageDataset
import os
from tqdm import tqdm

''' Storing Latents
In this file, we will simply make a function,
that given a VAE, can run every single instance
of the dataset, and save the representations in 
a file.

We specify a batch-size if we want to
try to do more in parallel.
'''

def store_latents(dataset,path,model,batch_size=16):
    # takes entire dataset and maps to 
    # a specific path for latent images.
    # returns the path of the representations
    model.eval()# turn off training!
    latent_outputs = []
    full_path = os.path.join(path,'latent_storage.pt')
    
    # creating folder
    if not os.path.exists(path):
        os.makedirs(path)
    
    # if already there
    if os.path.exists(full_path):
        print("Mapping already exists, skipping latent-storage")
       
    # creating file with latents 
    else: # NEED for cuda memory
        with torch.no_grad(): 
            # take all images.
            ra = tqdm(range(len(dataset) // batch_size))
            for i in ra:
                # appending latent-outputs
                # passing-thru labels.
                index = i*batch_size
                batch = torch.stack([dataset[index+i][0] for i in range(batch_size)],dim=0).to('cuda')
                output = model(batch).chunk(batch_size,dim=0)
                for i in range(batch_size):
                    latent_outputs.append(output[i][0])
                
            save = [latent_outputs]
            torch.save(save,full_path)

