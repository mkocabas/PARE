import math
import torch
import numpy as np
import torch.nn as nn

from ...core.config import SMPL_MEAN_PARAMS
from ...utils.geometry import rot6d_to_rotmat

BN_MOMENTUM = 0.1


class HMR_NP_Head(nn.Module):
    def __init__(
            self,
            num_input_features,
            smpl_mean_params=SMPL_MEAN_PARAMS,
            estimate_var=False,
            use_separate_var_branch=False,
            uncertainty_activation='',
            backbone='resnet50',
            use_cam_feats=False,
    ):
        super(HMR_NP_Head, self).__init__()

        npose = 14 * 3
        self.npose = npose
        # self.estimate_var = estimate_var
        # self.use_separate_var_branch = use_separate_var_branch
        # self.uncertainty_activation = uncertainty_activation
        self.backbone = backbone
        self.num_input_features = num_input_features
        # self.use_cam_feats = use_cam_feats

        # if use_cam_feats:
        #     num_input_features += 7 # 6d rotmat + vfov

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(num_input_features, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()

        self.decpose = nn.Linear(1024, npose)
        # self.decshape = nn.Linear(1024, 10)
        # self.deccam = nn.Linear(1024, 3)

        # nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        # nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        # nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        if self.backbone.startswith('hrnet'):
            self.downsample_module = self._make_head()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def _make_head(self):
        # downsampling modules
        downsamp_modules = []
        for i in range(3):
            in_channels = self.num_input_features
            out_channels = self.num_input_features

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)

        downsamp_modules = nn.Sequential(*downsamp_modules)

        return downsamp_modules

    def forward(
            self,
            features,
            init_pose=None,
            init_shape=None,
            init_cam=None,
            cam_rotmat=None,
            cam_vfov=None,
            n_iter=3
    ):
        # if self.backbone.startswith('hrnet'):
        #     features = self.downsample_module(features)

        batch_size = features.shape[0]

        # if init_pose is None:
        #     init_pose = self.init_pose.expand(batch_size, -1)
        # if init_shape is None:
        #     init_shape = self.init_shape.expand(batch_size, -1)
        # if init_cam is None:
        #     init_cam = self.init_cam.expand(batch_size, -1)

        xf = self.avgpool(features)
        xf = xf.view(xf.size(0), -1)

        pred_pose = init_pose
        # pred_shape = init_shape
        # pred_cam = init_cam
        for i in range(n_iter):
            # if self.use_cam_feats:
            #     xc = torch.cat([xf, pred_pose, pred_shape, pred_cam,
            #                     rotmat_to_rot6d(cam_rotmat), cam_vfov.unsqueeze(-1)], 1)
            # else:
            xc = xf # torch.cat([xf, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)

            pred_pose = self.decpose(xc) # + pred_pose
            # pred_shape = self.decshape(xc) + pred_shape
            # pred_cam = self.deccam(xc) + pred_cam

        # pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        output = {
            # 'pred_pose': pred_rotmat,
            # 'pred_cam': pred_cam,
            # 'pred_shape': pred_shape,
            # 'pred_pose_6d': pred_pose,
            'smpl_joints3d': pred_pose.reshape(batch_size, -1, 3)
        }

        # if self.estimate_var:
        #     output.update({
        #         'pred_pose_var': torch.cat([pred_pose, pred_pose_var], dim=1),
        #         'pred_shape_var': torch.cat([pred_shape, pred_shape_var], dim=1),
        #     })

        return output


from ..layers.contrib import adf


def keep_variance(x, min_variance):
    return x + min_variance


class hmr_head_adf_dropout(nn.Module):
    def __init__(
            self,
            num_input_features,
            smpl_mean_params=SMPL_MEAN_PARAMS,
            p=0.2,
            min_variance=1e-3,
            noise_variance=1e-3,
    ):
        super(hmr_head_adf_dropout, self).__init__()

        self.keep_variance_fn = lambda x: keep_variance(x, min_variance=min_variance)
        self._noise_variance = noise_variance

        npose = 24 * 6
        self.avgpool = adf.AvgPool2d(keep_variance_fn=self.keep_variance_fn) # nn.AvgPool2d(7, stride=1)
        self.fc1 = adf.Linear(num_input_features + npose + 13, 1024, keep_variance_fn=self.keep_variance_fn) # nn.Linear(num_input_features + npose + 13, 1024)
        self.drop1 = adf.Dropout(p=p, keep_variance_fn=self.keep_variance_fn) # nn.Dropout()
        self.fc2 = adf.Linear(1024, 1024, keep_variance_fn=self.keep_variance_fn) # nn.Linear(1024, 1024)
        self.drop2 = adf.Dropout(p=p, keep_variance_fn=self.keep_variance_fn) # nn.Dropout()
        self.decpose = adf.Linear(1024, npose, keep_variance_fn=self.keep_variance_fn) # nn.Linear(1024, npose)
        self.decshape = adf.Linear(1024, 10, keep_variance_fn=self.keep_variance_fn) # nn.Linear(1024, 10)
        self.deccam = adf.Linear(1024, 3, keep_variance_fn=self.keep_variance_fn) # nn.Linear(1024, 3)
        nn.init.xavier_uniform_(self.decpose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.decshape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.deccam.weight, gain=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        mean_params = np.load(smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)

    def forward(
            self,
            features,
            init_pose=None,
            init_shape=None,
            init_cam=None,
            n_iter=3
    ):
        batch_size = features[0].shape[0]

        if init_pose is None:
            init_pose = self.init_pose.expand(batch_size, -1)
        if init_shape is None:
            init_shape = self.init_shape.expand(batch_size, -1)
        if init_cam is None:
            init_cam = self.init_cam.expand(batch_size, -1)

        # inputs_mean = features
        # inputs_variance = torch.zeros_like(inputs_mean) + self._noise_variance
        # x = feat inputs_mean, inputs_variance
        xf = self.avgpool(*features, 7)
        xf_mean = xf[0].view(xf[0].size(0), -1)
        xf_var = xf[1].view(xf[1].size(0), -1)
        xf = xf_mean, xf_var

        pred_pose = init_pose, torch.zeros_like(init_pose) + self._noise_variance
        pred_shape = init_shape, torch.zeros_like(init_shape) + self._noise_variance
        pred_cam = init_cam, torch.zeros_like(init_cam) + self._noise_variance
        for i in range(n_iter):
            xc_mean = torch.cat([xf[0], pred_pose[0], pred_shape[0], pred_cam[0]], 1)
            xc_var = torch.cat([xf[1], pred_pose[1], pred_shape[1], pred_cam[1]], 1) # torch.zeros_like(xc_mean) + self._noise_variance
            xc = xc_mean, xc_var  # torch.zeros_like(xc) + self._noise_variance
            xc = self.fc1(*xc)
            xc = self.drop1(*xc)
            xc = self.fc2(*xc)
            xc = self.drop2(*xc)
            pred_pose = self.decpose(*xc)[0] + pred_pose[0], self.decpose(*xc)[1] + pred_pose[1]
            pred_shape = self.decshape(*xc)[0] + pred_shape[0], self.decshape(*xc)[1] + pred_shape[1]
            pred_cam = self.deccam(*xc)[0] + pred_cam[0], self.deccam(*xc)[1] + pred_cam[1]

        pred_rotmat = rot6d_to_rotmat(pred_pose[0]).view(batch_size, 24, 3, 3)
        pred_pose_var = pred_pose[1].reshape(-1, 24, 6)

        output = {
            'pred_pose': pred_rotmat,
            'pred_cam': pred_cam[0],
            'pred_shape': pred_shape[0],
            'pred_pose_var': pred_pose_var,
            'pred_cam_var': pred_cam[1],
            'pred_shape_var': pred_shape[1],
        }

        return output