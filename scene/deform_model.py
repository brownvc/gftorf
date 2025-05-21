import os
import torch
from utils.time_utils import DeformNetwork
from utils.general_utils import get_expon_lr_func
from arguments import ModelParams

class DeformModel:
    def __init__(self, args : ModelParams):
        kwargs = {
            'D': args.D, 
            'W': args.W, 
            'xyz_multires': args.xyz_multires,
            't_multires': args.t_multires,
            'sh_degree': args.sh_degree
        }
        self.deform = DeformNetwork(**kwargs).cuda()
        self.deform.initialize_weights(args)

    def training_setup(self, training_args, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

        l = [
            {'params': list(self.deform.parameters()),
             'lr': training_args.deform_lr_init,
             "name": "deform"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(lr_init=training_args.deform_lr_init,
                                                       lr_final=training_args.deform_lr_final,
                                                       lr_delay_mult=training_args.position_lr_delay_mult,
                                                       max_steps=(training_args.position_lr_max_steps-training_args.warm_up))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.deform.state_dict(), path)

    def load(self, path):
        self.deform.load_state_dict(torch.load(path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr