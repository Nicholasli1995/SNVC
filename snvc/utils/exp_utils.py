"""
A python class managing experimental configurations and logging.
"""

import os
import sys

from .logger_utils import colorlogger
from tensorboardX import SummaryWriter 

class Experimenter:
    def __init__(self, model_dir, cfg_path=None):
        self.model_dir = model_dir
        self.cfg_path = cfg_path

        if self.cfg_path:
            # if a config file is provided, update the config file in the model 
            # directory with the provided config if needed
            self.init_with_cfg_path()
        else:
            # if cfg_path is None, there should be a config file in the model directory
            assert os.path.exists('{}/save_config.py'.format(self.model_dir)), \
                'Found no config in the model_dir: {}'.format(self.model_dir)
        
        # load the configuration file
        sys.path.insert(0, self.model_dir)
        from save_config import cfg
        self.cfg = cfg

    @property
    def config(self):
        return self.cfg

    @property
    def logger(self):
        if not hasattr(self, '_logger'):
            print('Log -->', os.path.join(self.model_dir, 'training.log'))
            self._logger = colorlogger(self.model_dir)

        return self._logger

    @property
    def writer(self):
        if not hasattr(self, '_writer'):
            self.tensorboard_dir = os.path.join(self.model_dir, 'tensorboard')
            print('Tensorboard -->', self.tensorboard_dir)
            self._writer = SummaryWriter(self.tensorboard_dir)

        return self._writer
    
    def init_with_cfg_path(self):
        if not os.path.exists(self.cfg_path):
            raise ValueError('The configuration file {} does not exist'.format(self.cfg_path))
            
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        
        # the configuration file will be saved to save_path
        save_path = os.path.join(self.model_dir, "save_config.py")
        if os.path.normpath(self.cfg_path) == os.path.normpath(save_path):
            print('Using the configuration file at {}'.format(save_path))
            return

        # if an old saved file exists, create a backup
        if os.path.exists(save_path):
            old_file = os.path.join(self.model_dir, "save_config.py")
            old_file_tmp = os.path.join(self.model_dir, "save_config.py.tmp")
            os.system('mv {} {}'.format(old_file, old_file_tmp))
            
        # save the new configuration file
        backup_path = os.path.join(self.model_dir, "save_config.py")
        os.system('cp {} {}'.format(self.cfg_path, backup_path))
        print('The configuration file {} is saved to {}'.format(self.cfg_path, backup_path))       