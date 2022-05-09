import time
import torch

from .temporal_smplify import TemporalSMPLify
from .smplify import SMPLify
from ..utils.geometry import rotation_matrix_to_angle_axis, batch_rodrigues


def smplify_runner(
        pred_rotmat,
        pred_betas,
        pred_cam,
        j2d,
        device,
        batch_size,
        lr=1.0,
        opt_steps=1,
        use_lbfgs=True,
        pose2aa=True,
        is_video=False,
):
    if is_video:
        smplify = TemporalSMPLify(
            step_size=lr,
            batch_size=batch_size,
            num_iters=opt_steps,
            focal_length=5000.,
            use_lbfgs=use_lbfgs,
            device=device,
            # max_iter=10,
        )
    else:
        smplify = SMPLify(
            step_size=lr,
            batch_size=batch_size,
            num_iters=opt_steps,
            focal_length=5000.,
            use_lbfgs=use_lbfgs,
            device=device,
            # max_iter=10,
        )
    # Convert predicted rotation matrices to axis-angle
    if pose2aa:
        pred_pose = rotation_matrix_to_angle_axis(pred_rotmat.detach().reshape(-1, 3, 3)).reshape(batch_size, -1)
    else:
        pred_pose = pred_rotmat

    # Calculate camera parameters for smplify
    pred_cam_t = torch.stack([
        pred_cam[:, 1], pred_cam[:, 2],
        2 * 5000 / (224 * pred_cam[:, 0] + 1e-9)
    ], dim=-1)

    gt_keypoints_2d_orig = j2d
    # Before running compute reprojection error of the network
    opt_joint_loss = smplify.get_fitting_loss(
        pred_pose.detach(), pred_betas.detach(),
        pred_cam_t.detach(),
        0.5 * 224 * torch.ones(batch_size, 2, device=device),
        gt_keypoints_2d_orig).mean(dim=-1)

    best_prediction_id = torch.argmin(opt_joint_loss).item()
    pred_betas = pred_betas[best_prediction_id].unsqueeze(0)
    # pred_betas = pred_betas[best_prediction_id:best_prediction_id+2] # .unsqueeze(0)
    # top5_best_idxs = torch.topk(opt_joint_loss, 5, largest=False)[1]
    # breakpoint()

    start = time.time()
    # Run SMPLify optimization initialized from the network prediction
    # new_opt_vertices, new_opt_joints, \
    # new_opt_pose, new_opt_betas, \
    # new_opt_cam_t, \
    output, new_opt_joint_loss = smplify(
        pred_pose.detach(), pred_betas.detach(),
        pred_cam_t.detach(),
        0.5 * 224 * torch.ones(batch_size, 2, device=device),
        gt_keypoints_2d_orig,
    )
    new_opt_joint_loss = new_opt_joint_loss.mean(dim=-1)
    # smplify_time = time.time() - start
    # print(f'Smplify time: {smplify_time}')
    # Will update the dictionary for the examples where the new loss is less than the current one
    update = (new_opt_joint_loss < opt_joint_loss)

    new_opt_vertices = output['verts']
    new_opt_cam_t = output['theta'][:,:3]
    new_opt_pose = output['theta'][:,3:75]
    new_opt_betas = output['theta'][:,75:]
    new_opt_joints3d = output['kp_3d']

    new_opt_pose = batch_rodrigues(new_opt_pose.reshape(-1, 3)).reshape(batch_size, 24, 3, 3)

    return_val = [
        update, new_opt_vertices.cpu(), new_opt_cam_t.cpu(),
        new_opt_pose.cpu(), new_opt_betas.cpu(), new_opt_joints3d.cpu(),
        new_opt_joint_loss, opt_joint_loss,
    ]

    return return_val