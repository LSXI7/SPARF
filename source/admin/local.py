import os


# 只用修改1，2，3，5的数据
class EnvironmentSettings:
    def __init__(self, data_root='', debug=False):
        self.workspace_dir = '/data/xyjiang/NeRF/sparf/output/llff_try/'  # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data/xyjiang/NeRF/sparf/output/llff_try/tensorboard/'  # Directory for tensorboard files.
        self.log_dir = '/data/xyjiang/NeRF/sparf/output/llff_try/log/'
        self.pretrained_networks = self.workspace_dir  # Directory for saving other models pre-trained networks
        self.eval_dir = '/data/xyjiang/NeRF/sparf/output/llff_try/eval_dir'  # Base directory for saving the evaluations.
        self.llff = '/data/xyjiang/NeRF/datasets4Nerf/nerf_llff_data/'
        self.dtu = '/data/xyjiang/NeRF/datasets4Nerf/DTU_for_sparf/rs_dtu_4/DTU/'
        self.dtu_depth = '/data/xyjiang/NeRF/datasets4Nerf/DTU_for_sparf/'
        self.dtu_mask = '/data/xyjiang/NeRF/datasets4Nerf/DTU_for_sparf/submission_data/idrmasks/'
        self.replica = '/data/xyjiang/NeRF/datasets4Nerf/Replica/Replica/'
