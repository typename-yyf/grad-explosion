import torch
import numpy as np
import random
import json
import os
from accelerate import Accelerator

def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

NONE = 0
LOSS_SPIKE_DETECTED = 1
LARGE_GRAD_DETECTED = 2

class explosion_detector:

    '''
    Constructor function
    
    Parameters:
     model - 
     accelerator - 
     seed - random seed for result reproducibility
     alpha - decay coefficient for the average gradient norm
     beta - threshold for detecting loss spikes / loss explosion
     gamma - threshold for detecting large gradients
    
    Returns:
     None
    
    Raises:
     None
    '''
    def __init__(
        self,
        model: torch.nn.Module,
        accelerator: Accelerator,
        seed: int=0,
        alpha: float=0.9,
        beta: float=4.0,
        gamma: float=3.0
    ):
        _set_seed(seed)
        self.model = model
        self.accelerator = accelerator
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.step = -1
        self.history_loss_list_len = 20

        self._history_loss_list = []
        self._history_grad_norm = 0

    '''
    This function is used for saving checkpoints.
    
    Parameters:
     path - path for saving checkpoints
     step - step corresponding to the checkpoint. default: the automatically recorded step
     dataloder_offset - 
    
    Returns:
     None
    
    Raises:
     None
    '''
    def save(
        self, 
        path: str="model",
        step: int=-1,
        dataloder_offset: int=0
    ) -> None:
        try:
            os.mkdir(path)
        except:
            pass

        self.accelerator.save_state(path + "/pth")
        fd = open(path + "/state.json", "w")
        d = {
            "seed": self.seed,
            "step": step if step >= 0 else self.step,
            "dataloder_offset": dataloder_offset,
        }
        json.dump(d, fd)
        fd.close()
    
    '''
    This function is used for loading model from checkpoints.
    
    Parameters:
     path - path of the saved checkpoints
    
    Returns:
     None
    
    Raises:
     None
    '''
    def load(
        self,
        path: str="model"
    ) -> None:
        self.accelerator.load_state(path + "/pth")
        fd = open(path + "/state.json", "r")
        d = json.load(fd)
        self.seed = int(d["seed"])
        self.step = int(d["step"])
        self.dataloder_offset = int(d["dataloder_offset"])
        fd.close()


    '''
    This function is called after each backward pass, and its invocation interval can be arbitrary. 
    It can be called once per epoch, per step, or every 100 steps, for example. 
    It is used to detect the presence of loss spikes or gradient explosions.
    
    Parameters:
     loss - path of the saved checkpoints
     decreased_loss - Ture if the loss decreases during the training
    
    Returns:
     NONE - model is in a normal training state
     LOSS_SPIKE_DETECTED - 
     LARGE_GRAD_DETECTED - 
    
    Raises:
     None
    '''
    def apply(
        self, 
        loss: float, 
        decreased_loss: bool=True,
    ) -> int:
        self.step += 1
        rv = 0

        if len(self.history_loss_list) > 0:
            mean = np.mean(self.history_loss_list)
            std = np.std(self.history_loss_list)
            if decreased_loss:
                if loss > mean + self.beta * std:
                    rv |= LOSS_SPIKE_DETECTED
            else:
                if loss < mean - self.beta * std:
                    rv |= LOSS_SPIKE_DETECTED
        
        if len(self.history_loss_list) >= self.history_loss_list_len:
            self.history_loss_list.pop(0)
        self.history_loss_list.append(loss)


        grad_norm = 0
        for p in self.model.parameters():
            grad_norm += p.gard.norm().item()
        
        if self._history_grad_norm == 0:
            self._history_grad_norm = grad_norm
        else:
            if self._history_grad_norm * self.gamma < grad_norm:
                rv |= LARGE_GRAD_DETECTED

            self._history_grad_norm = self.gamma * self._history_grad_norm + \
                                      (1 - self.gamma) * grad_norm
            
        return rv
        

        
