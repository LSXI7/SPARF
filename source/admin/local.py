import os

# 只用修改1，2，3，5的数据
class EnvironmentSettings:
    def __init__(self, data_root='', debug=False):
        self.workspace_dir = '/data/xyjiang/NeRF/sparf/try/'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data/xyjiang/NeRF/sparf/try/tensorboard/'    # Directory for tensorboard files.
        self.log_dir = '/data/xyjiang/NeRF/sparf/try/log/'
        self.pretrained_networks = self.workspace_dir    # Directory for saving other models pre-trained networks
        self.eval_dir = '/data/xyjiang/NeRF/sparf/try/eval_dir'    # Base directory for saving the evaluations.
        self.llff = '/data/xyjiang/NeRF/datasets4Nerf/nerf_llff_data/'
        self.dtu = '/data/xyjiang/NeRF/datasets4Nerf/DTU_for_sparf/rs_dtu_4/DTU/'
        self.dtu_depth = '/data/xyjiang/NeRF/datasets4Nerf/DTU_for_sparf/'
        self.dtu_mask = '/data/xyjiang/NeRF/datasets4Nerf/DTU_for_sparf/submission_data/idrmasks/'
        self.replica = ''


#import os

#class EnvironmentSettings:
#    def __init__(self, data_root='', debug=False):
#        self.workspace_dir = '/home/xyjiang/sparf/try/'    # Base directory for saving network checkpoints.
#        self.tensorboard_dir = '/home/xyjiang/sparf/tensorboard/'    # Directory for tensorboard files.
#        self.log_dir = '/home/xyjiang/sparf/log/'
#        self.pretrained_networks = self.workspace_dir    # Directory for saving other models pre-trained networks
#        self.eval_dir = '/home/xyjiang/sparf/eval_dir/'    # Base directory for saving the evaluations. 
#        self.llff = ''
#        self.dtu = '/home/xyjiang/sparf/data/DTU/pixel_nerf_data/rs_dtu_4/DTU/'
#        self.dtu_depth = '/home/xyjiang/sparf/data/DTU/'
#        self.dtu_mask = '/home/xyjiang/sparf/data/DTU/submission_data/idrmasks/'
#        self.replica = ''
