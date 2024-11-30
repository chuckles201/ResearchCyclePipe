# Resarch Cycle Pipeline Repo


Here is where I will be saving the pipeline I use for loading data and training neural networks.

Having to write code to load a dataset *every time* I create a research project is just unsustainable, so I aim here to create a pipeline for downloading, loading data, and training *so I can focus on the architecture and design of neural networks themselves, not the pipeline/data management*.

Here is what I'll be implementing:
1. DataSet class to download data/labels, visualize, and easily convert to batches for training and eval.
2. Ability to store latent representations/mappings (for latent-diffusion models), and use these instead of the images 
3. Be able to visualize the training of the model (and maybe expiriments/parameters?)


## What I have so far

##### ***dataloader.py***
- Load raw data of images, for easier use with custom 
classes
- Creates custom dataset, where we can acess latent-folder
everything is just by number, and no labels are used
(specific to generative/self-supervised models)

##### ***make_latents.py***
- stores all latent tensors into a csv file,
which have indices matching that of the corresp.
image.
- Integrated into data-class so can easily be acessed.