import sys
from collections import OrderedDict

import data
import evaluation.group_evaluator as group_evaluator
import util.util as util
from options.train_options import TrainOptions
from trainers import get_trainer
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
import os
from torchvision.utils import save_image
from torchvision import transforms
os.environ["WANDB_MODE"] = "dryrun"

import torch
import tqdm
import torch.nn.functional as F

import numpy as np
import time
import random

def fix_seed(seed):
    """Fixes the seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    

# os.makedirs('images/', exist_ok=True)
# os.makedirs('saved_models/', exist_ok=True)

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# saved folder with current date and time
SAVED_IMAGES_DIR = f'{opt.exp_name}/image_{time.strftime("%Y-%m-%d_%H-%M-%S")}'

os.makedirs(SAVED_IMAGES_DIR, exist_ok=True)

fix_seed(opt.seed)

# load the dataset
dataloader = data.create_dataloader(opt)
test_dataloader = data.create_dataloader(util.copyconf(opt, phase="test", batch_size=1))

# create trainer for our model
trainer = get_trainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader.dataset))

# create tool for visualization
visualizer = Visualizer(opt)

# create evaluator for evaluation
evaluator = group_evaluator.GroupEvaluator(opt)

def save_images_as_png(input_tensor, target_tensor, generated_tensor, step, epoch=None, global_generated=None):
    # os.makedirs('images/', exist_ok=True)
    os.makedirs(SAVED_IMAGES_DIR, exist_ok=True)

    # 1) Process the input tensor: take channel 8, repeat -> 3 channels
    # Clamp input to [-1,1] -> [0,1]
    if input_tensor.shape[1] > 3:
        input_slice = input_tensor[:, -2:-1, :, :].clamp(0, 1)
        # input_slice = (input_slice + 1) / 2
        input_3ch = input_slice.repeat(1, 3, 1, 1)

    else:
        input_3ch = input_tensor.clamp(0, 1)

    # 2) Clamp target and generated to [-1,1] -> [0,1]
    target_tensor = (target_tensor.clamp(-1, 1) + 1) / 2
    generated_tensor = (generated_tensor.clamp(-1, 1) + 1) / 2

    if global_generated is not None:
        global_generated = (global_generated.clamp(-1, 1) + 1) / 2

    # Move to CPU for processing/saving
    input_3ch = input_3ch.cpu()
    target_tensor = target_tensor.cpu()
    generated_tensor = generated_tensor.cpu()

    if global_generated is not None:
        global_generated = global_generated.cpu()

    batch_size = input_3ch.shape[0]

    # Collect resized images for each category in lists
    in_list = []
    tar_list = []
    gen_list = []
    if global_generated is not None:
        global_gen_list = []

    for b in range(batch_size):
        # if current image is great than 1024x1024then resize
        
        in_img = input_3ch[b] 
        tar_img = target_tensor[b]
        gen_img = generated_tensor[b]
        # from pdb import set_trace; set_trace()
        if global_generated is not None:
            global_gen_img = global_generated[b]

        in_list.append(in_img)
        tar_list.append(tar_img)
        gen_list.append(gen_img)
        if global_generated is not None:
            global_gen_list.append(global_gen_img)

    # Concatenate each category horizontally: shape -> [3, 256, batch_size*256]
    in_cat = torch.cat(in_list, dim=2)
    tar_cat = torch.cat(tar_list, dim=2)
    gen_cat = torch.cat(gen_list, dim=2)
    if global_generated is not None:
        global_gen_cat = torch.cat(global_gen_list, dim=2)

    # Finally, stack these rows (input, target, generated) vertically
    # shape -> [3, 3*256, batch_size*256]
    # combined = torch.cat([in_cat, tar_cat, gen_cat], dim=1)
    if global_generated is not None:
        combined = torch.cat([in_cat, tar_cat, gen_cat, global_gen_cat], dim=1)
    else:
        combined = torch.cat([in_cat, tar_cat, gen_cat], dim=1)

    # Save one single PNG per iteration/step with the entire batch
    # saved_png = f'images/combined_step{step}.png' if epoch is None \
    #     else f'images/combined_epoch{epoch}_step{step}.png'

    saved_png = f'{SAVED_IMAGES_DIR}/combined_step{step}.png' if epoch is None \
        else f'{SAVED_IMAGES_DIR}/combined_epoch{epoch}_step{step}.png'
    
    save_image(combined, f'{saved_png}')#, nrow=batch_size, normalize=False)




for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    # using tqdm
    # progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch}/{opt.niter + opt.niter_decay}")
    # for i, data_i in enumerate(progress_bar):
        # progress_bar.set_postfix({"epoch": epoch, "iter": i})
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
    
        
        iter_counter.record_one_iteration()

        # train generator
        if i % opt.D_steps_per_G == 0:
            for _ in range(opt.num_G_steps):
                trainer.run_generator_one_step(data_i, iter=i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(
                epoch,
                iter_counter.epoch_iter,
                losses,
                iter_counter.time_per_iter, trainer.old_lr
            )
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        # from pdb import set_trace; set_trace()

        
        generated = trainer.get_latest_generated()
        # generated = F.interpolate(generated, size=data_i['input'].shape[2:], mode='bilinear', align_corners=True)
        
        # if isinstance(generated, tuple):
        #     global_generated = generated[1]
        #     generated = generated[0]
            
        # generated = reassemble(generated)

        

        if iter_counter.needs_displaying():
            local_images = data_i['input_patches']
            output_patches = data_i['target_patches'].squeeze(1)

            save_images_as_png(
                local_images,
                output_patches,
                generated,
                iter_counter.total_steps_so_far,
                epoch=epoch,
                global_generated=None,
            )

        if iter_counter.needs_saving():
            print(
                f'saving the latest model (epoch {epoch}, total_steps {iter_counter.total_steps_so_far})'
            )
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
        print(
            f'saving the model at the end of epoch {epoch}, iters {iter_counter.total_steps_so_far}'
        )
        trainer.save(f'{epoch}')
        trainer.save(epoch)

print('Training was successfully finished.')

