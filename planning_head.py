import numpy as np
import torch
import torch.nn.functional as F


class RADModel:
    def __init__(self, x_anchor=61, y_anchor=61, fut_ts=6):
        self.x_anchor = x_anchor
        self.y_anchor = y_anchor
        self.fut_ts = fut_ts
        self.total_anchors = self.x_anchor * self.y_anchor
        self.use_top1_sample = True

        plan_anchors_path = "./data/traj_anchor_05s_3721.npy"
        self.plan_anchors = torch.from_numpy(
            np.load(plan_anchors_path).astype(np.float32)
        )
        plan_anchors_yaw_path = "./data/traj_anchor_yaw_05s_3721.npy"
        self.plan_anchors_yaw = torch.from_numpy(
            np.load(plan_anchors_yaw_path).astype(np.float32)
        )
        # Some anchors correspond to physically infeasible actions and need to be masked
        plan_anchors_mask_path = "./data/traj_anchor_mask_05s_3721.npy"
        self.plan_anchors_mask = torch.from_numpy(
            np.load(plan_anchors_mask_path).reshape(-1)
        )

    def loss_discrete_plan_x(self, pred, gt):
        return F.cross_entropy(pred, gt)

    def loss_discrete_plan_y(self, pred, gt):
        return F.cross_entropy(pred, gt)

    def val(self, preds_dicts):
        # Simulated history trajectory data (for validation/testing only)
        clip_datas = {
            # A fake timestamp placeholder
            "timestamp": torch.tensor(000000),
            # Simulated odometry information representing past trajectory.
            # Each row corresponds to a past frame with (x, y, yaw).
            "odo_info": torch.tensor([
                [0.0, 0.0, 0.0],  # frame t-2
                [0.0, 0.0, 0.0],  # frame t-1
                [0.0, 0.0, 0.0]   # current frame
            ])
        }

        outputs_ego_x_action = preds_dicts['outputs_ego_x_action'].squeeze(-1)
        outputs_ego_y_action = preds_dicts['outputs_ego_y_action'].squeeze(-1)
        outputs_ego_x_action_ordinal = F.softmax(outputs_ego_x_action,dim=-1)
        outputs_ego_y_action_ordinal = F.softmax(outputs_ego_y_action,dim=-1)

        if self.use_top1_sample:
            selected_anchor_idx_x = outputs_ego_x_action_ordinal.argmax(dim=-1)[0]
            selected_anchor_idx_y = outputs_ego_y_action_ordinal.argmax(dim=-1)[0]
        else:
            selected_anchor_idx_x = torch.multinomial(outputs_ego_x_action_ordinal, num_samples=1)[0][0]
            selected_anchor_idx_y = torch.multinomial(outputs_ego_y_action_ordinal, num_samples=1)[0][0]

        preds_dicts['outputs_ego_cls_action'] = torch.mul(
            outputs_ego_x_action_ordinal.unsqueeze(2),
            outputs_ego_y_action_ordinal.unsqueeze(1)
        ).reshape(1, self.x_anchor * self.y_anchor, 1)

        selected_anchor_idx = selected_anchor_idx_x * self.y_anchor + selected_anchor_idx_y
        selected_anchor = self.plan_anchors[selected_anchor_idx]
        x_100ms = selected_anchor[1, 0].cpu().item()
        y_100ms = selected_anchor[1, 1].cpu().item()
        yaw_100ms = self.plan_anchors_yaw[selected_anchor_idx].cpu().item()

        RT_action = np.array([
            [np.cos(yaw_100ms), -np.sin(yaw_100ms), 0, x_100ms],
            [np.sin(yaw_100ms),  np.cos(yaw_100ms), 0, y_100ms],
            [0,                 0,                  1, 0],
            [0,                 0,                  0, 1]
        ])

        print("RT_action:\n", RT_action)

        odo_info_next = clip_datas["odo_info"].clone().numpy()
        odo_info_next[:-1] = odo_info_next[1:]
        cur_x_global, cur_y_global, cur_yaw_global = odo_info_next[-1]

        action_dis = np.sqrt(x_100ms**2 + y_100ms**2)
        angle = np.arctan2(y_100ms, x_100ms)

        action_dx_global = action_dis * np.cos(angle + cur_yaw_global)
        action_dy_global = action_dis * np.sin(angle + cur_yaw_global)

        normalized_yaw = (cur_yaw_global + yaw_100ms) % (2 * np.pi)
        if normalized_yaw >= np.pi:
            normalized_yaw -= 2 * np.pi

        odo_info_next[-1, 0] = cur_x_global + action_dx_global
        odo_info_next[-1, 1] = cur_y_global + action_dy_global
        odo_info_next[-1, 2] = normalized_yaw

        print("Updated odo_info:\n", odo_info_next)

    def compute_rl_loss(self, preds_dicts, rl_metainfo):
        """
        Example RL planning loss computation function

        Args:
            preds_dicts (dict): Model prediction results, including ego future action predictions.
            rl_metainfo (dict): Reinforcement learning metadata, including advantage values and other rollout information.
        
        Returns:
            loss_dict (dict): A dictionary containing RL planning loss components.
        """

        batch = preds_dicts["outputs_ego_x_action"].shape[0]
        rl_value_function = preds_dicts["rl_value_function"]
        outputs_ego_x_action = preds_dicts['outputs_ego_x_action'].squeeze(-1)
        outputs_ego_y_action = preds_dicts['outputs_ego_y_action'].squeeze(-1)
        outputs_ego_x_action_ordinal = F.softmax(outputs_ego_x_action,dim=-1)
        outputs_ego_y_action_ordinal = F.softmax(outputs_ego_y_action,dim=-1)

        dynamic_collision_advantage = torch.tensor(
            rl_metainfo['dynamic_collision_advantage'], device=outputs_ego_x_action.device
        )
        static_collision_advantage = torch.tensor(
            rl_metainfo['static_collision_advantage'], device=outputs_ego_x_action.device
        )
        distance_deviation_advantage = torch.tensor(
            rl_metainfo['distance_deviation_advantage'], device=outputs_ego_x_action.device
        )
        angle_deviation_advantage = torch.tensor(
            rl_metainfo['angle_deviation_advantage'], device=outputs_ego_x_action.device
        )
        old_collision_value = torch.tensor(
            rl_metainfo['old_collision_value'], device=outputs_ego_x_action.device
        )

        collision_advantage = dynamic_collision_advantage + static_collision_advantage
        static_coll_deviation_advantage = static_collision_advantage + distance_deviation_advantage + angle_deviation_advantage
        b_collision_return = dynamic_collision_advantage + old_collision_value  

        newprobs_x, newprobs_y = [], []
        pseudo_odo_pred_probs_x, pseudo_odo_pred_probs_y = [], []
        neg_odo_pred_probs_x, neg_odo_pred_probs_y = [], []
        regular_odo_pred_probs_x, regular_odo_pred_probs_y = [], []

        old_prob_x, old_prob_y = [], []

        for i in range(batch):
            old_selected_anchor_idx_x = rl_metainfo['selected_anchor_idx_x']
            old_selected_anchor_idx_y = rl_metainfo['selected_anchor_idx_y']
            old_outputs_ego_x = torch.tensor(
                rl_metainfo['outputs_ego_x_action_pred'][i, old_selected_anchor_idx_x],
                device=outputs_ego_x_action.device
            )
            old_outputs_ego_y = torch.tensor(
                rl_metainfo['outputs_ego_y_action_pred'][i, old_selected_anchor_idx_y],
                device=outputs_ego_x_action.device
            )
            old_prob_x.append(old_outputs_ego_x[None])
            old_prob_y.append(old_outputs_ego_y[None])

            outputs_x = preds_dicts['outputs_ego_x_action'].squeeze(-1)
            outputs_y = preds_dicts['outputs_ego_y_action'].squeeze(-1)
            outputs_x_ordinal = F.softmax(outputs_x, dim=-1)
            outputs_y_ordinal = F.softmax(outputs_y, dim=-1)

            newprobs_i_x = outputs_x_ordinal[i, old_selected_anchor_idx_x][None]
            newprobs_i_y = outputs_y_ordinal[i, old_selected_anchor_idx_y][None]
            newprobs_x.append(newprobs_i_x)
            newprobs_y.append(newprobs_i_y)

            # action yaw
            distance_deviation = torch.tensor(rl_metainfo['distance_deviation'],device=outputs_ego_x_action.device,dtype=outputs_ego_x_action.dtype).detach()
            ego2match_yaw_degrees = torch.tensor(rl_metainfo['ego2match_yaw_degrees'],device=outputs_ego_x_action.device,dtype=outputs_ego_x_action.dtype).detach()
            first_collision_position = torch.tensor(rl_metainfo['first_collision_position'][i],device=outputs_ego_x_action.device,dtype=outputs_ego_x_action.dtype).detach()

            has_dynamic_collision = rl_metainfo['has_dynamic_collision']

            valid_anchors_y = torch.full((self.y_anchor,), False, dtype=torch.bool, device=outputs_ego_x_action.device)
            invalid_anchors_y = torch.full((self.y_anchor,), False, dtype=torch.bool, device=outputs_ego_x_action.device)
            valid_anchors_x = torch.full((self.x_anchor,), False, dtype=torch.bool, device=outputs_ego_x_action.device)
            invalid_anchors_x = torch.full((self.x_anchor,), False, dtype=torch.bool, device=outputs_ego_x_action.device)

            if ego2match_yaw_degrees < -0.0:  # Currently deviated to the left relative to the expert trajectory
                # Y-direction: actions that turn right relative to the previous action are encouraged
                # Y-direction: actions that turn left relative to the previous action are suppressed
                if old_selected_anchor_idx_x == 0:
                    # Special case at the starting point: only the current position is valid
                    valid_anchors_y[old_selected_anchor_idx_y] = True
                else:
                    # Non-starting point: positions after the current are valid (encourage right turn), before are invalid (suppress left turn)
                    valid_anchors_y[old_selected_anchor_idx_y+1:] = True
                    invalid_anchors_y[:old_selected_anchor_idx_y] = True

                # ========================== Dynamic collision adjustment (only X-direction) ==========================
                if abs(collision_advantage) > 1e-5 and has_dynamic_collision:
                    if first_collision_position[0] >= 0:
                        # Encourage deceleration, penalize acceleration
                        valid_anchors_x[:old_selected_anchor_idx_x] = True
                        invalid_anchors_x[old_selected_anchor_idx_x+1:] = True
                    else:
                        # Encourage acceleration, penalize deceleration
                        valid_anchors_x[old_selected_anchor_idx_x+1:] = True
                        invalid_anchors_x[:old_selected_anchor_idx_x] = True
                else:
                    valid_anchors_x[old_selected_anchor_idx_x] = True

            elif ego2match_yaw_degrees >= 0.0:  # Currently deviated to the right relative to the expert trajectory
                # Y-direction: actions that turn left relative to the previous action are encouraged
                # Y-direction: actions that turn right relative to the previous action are suppressed
                if old_selected_anchor_idx_x == 0:
                    # Special handling at the starting point: only the current position is valid
                    valid_anchors_y[old_selected_anchor_idx_y] = True
                else:
                    # Non-starting point: positions before the current are valid (encourage left turn), after are invalid (suppress right turn)
                    valid_anchors_y[:old_selected_anchor_idx_y] = True
                    invalid_anchors_y[old_selected_anchor_idx_y+1:] = True

                # ---------- Dynamic collision adjustment ----------
                if abs(collision_advantage) > 1e-5 and has_dynamic_collision:
                    if first_collision_position[0] >= 0:
                        # Encourage deceleration, penalize acceleration
                        valid_anchors_x[:old_selected_anchor_idx_x] = True
                        invalid_anchors_x[old_selected_anchor_idx_x+1:] = True
                    else:
                        # Encourage acceleration, penalize deceleration
                        valid_anchors_x[old_selected_anchor_idx_x+1:] = True
                        invalid_anchors_x[:old_selected_anchor_idx_x] = True
                else:
                    valid_anchors_x[old_selected_anchor_idx_x] = True

            old_selected_anchor_idx_x = rl_metainfo['selected_anchor_idx_x']
            old_selected_anchor_idx_y = rl_metainfo['selected_anchor_idx_y']

            pseudo_odo_pred_prob_x = outputs_ego_x_action_ordinal[i, valid_anchors_x]
            pseudo_odo_pred_prob_y = outputs_ego_y_action_ordinal[i, valid_anchors_y]
            neg_odo_pred_prob_x = outputs_ego_x_action_ordinal[i, invalid_anchors_x]
            neg_odo_pred_prob_y = outputs_ego_y_action_ordinal[i, invalid_anchors_y]

            pseudo_odo_pred_probs_x.append(pseudo_odo_pred_prob_x)
            pseudo_odo_pred_probs_y.append(pseudo_odo_pred_prob_y)
            neg_odo_pred_probs_x.append(neg_odo_pred_prob_x)
            neg_odo_pred_probs_y.append(neg_odo_pred_prob_y)

        pseudo_odo_pred_probs_x = torch.cat(pseudo_odo_pred_probs_x)
        pseudo_odo_pred_probs_y = torch.cat(pseudo_odo_pred_probs_y)
        neg_odo_pred_probs_x = torch.cat(neg_odo_pred_probs_x)
        neg_odo_pred_probs_y = torch.cat(neg_odo_pred_probs_y)

        newprobs_x = torch.cat(newprobs_x)
        newprobs_y = torch.cat(newprobs_y)
        old_prob_x = torch.cat(old_prob_x)
        old_prob_y = torch.cat(old_prob_y)

        new_collision_value = rl_value_function[:,0,0]
        ratio_x = newprobs_x / old_prob_x 
        ratio_y = newprobs_y / old_prob_y 

        clip_coef_x = 0.2
        clip_coef_y = 0.1
        # ============================= x ====================================
        pg_loss1_x = -collision_advantage * ratio_x
        pg_loss2_x = -collision_advantage * torch.clamp(ratio_x, 1 - clip_coef_x, 1 + clip_coef_x) 
        pg_loss_x = torch.max(pg_loss1_x, pg_loss2_x).sum()

        not_clamp_x = (ratio_x == torch.clamp(ratio_x, 1 - clip_coef_x, 1 + clip_coef_x))

        # # # collision
        dynamic_collision_p_weight = 1.0 if (dynamic_collision_advantage < 0.0 and has_dynamic_collision) else 0.0
        dynamic_collision_p_loss1_x = dynamic_collision_p_weight * (-dynamic_collision_advantage) * (-pseudo_odo_pred_probs_x) * not_clamp_x 
        dynamic_collision_p_loss2_x = dynamic_collision_p_weight * (-dynamic_collision_advantage) * (-pseudo_odo_pred_probs_x) * not_clamp_x
        dynamic_collision_p_loss_x = torch.min(dynamic_collision_p_loss1_x, dynamic_collision_p_loss2_x).sum()

        # collision_n_loss
        dynamic_collision_n_weight = 1.0 if (dynamic_collision_advantage < 0.0 and has_dynamic_collision) else 0.0
        dynamic_collision_n_loss1_x = dynamic_collision_n_weight * dynamic_collision_advantage * (-neg_odo_pred_probs_x) * not_clamp_x 
        dynamic_collision_n_loss2_x = dynamic_collision_n_weight * dynamic_collision_advantage * (-neg_odo_pred_probs_x) * not_clamp_x
        dynamic_collision_n_loss_x = torch.max(dynamic_collision_n_loss1_x, dynamic_collision_n_loss2_x).sum()

        # ============================= y ====================================
        pg_loss1_y =  -static_coll_deviation_advantage * ratio_y
        pg_loss2_y =  -static_coll_deviation_advantage * torch.clamp(ratio_y, 1 - clip_coef_y, 1 + clip_coef_y) 
        pg_loss_y = torch.max(pg_loss1_y, pg_loss2_y).sum()

        not_clamp_y = (ratio_y == torch.clamp(ratio_y, 1 - clip_coef_y, 1 + clip_coef_y))

        # deviation 
        deviation_distance_p_loss1_y = (-distance_deviation_advantage) * (-pseudo_odo_pred_probs_y) * not_clamp_y 
        deviation_distance_p_loss2_y = (-distance_deviation_advantage) * (-pseudo_odo_pred_probs_y) * not_clamp_y
        deviation_distance_p_loss_y = torch.min(deviation_distance_p_loss1_y, deviation_distance_p_loss2_y).sum()

        # deviation_n_loss
        deviation_distance_n_loss1_y = distance_deviation_advantage * (-neg_odo_pred_probs_y) * not_clamp_y 
        deviation_distance_n_loss2_y = distance_deviation_advantage * (-neg_odo_pred_probs_y) * not_clamp_y
        deviation_distance_n_loss_y = torch.max(deviation_distance_n_loss1_y, deviation_distance_n_loss2_y).sum()
        
        # # # collision
        collision_advantage_p_weight = 1.0 if static_collision_advantage < 0.0 else 0.0
        static_collision_p_loss1_y = collision_advantage_p_weight * (-static_collision_advantage) * (-pseudo_odo_pred_probs_y) * not_clamp_y
        static_collision_p_loss2_y = collision_advantage_p_weight * (-static_collision_advantage) * (-pseudo_odo_pred_probs_y) * not_clamp_y
        static_collision_p_loss_y = torch.min(static_collision_p_loss1_y, static_collision_p_loss2_y).sum()

        # collision_n_loss
        collision_advantage_n_weight = 1.0 if static_collision_advantage < 0.0 else 0.0
        static_collision_n_loss1_y = collision_advantage_n_weight * (static_collision_advantage) * (-neg_odo_pred_probs_y) * not_clamp_y
        static_collision_n_loss2_y = collision_advantage_n_weight * (static_collision_advantage) * (-neg_odo_pred_probs_y) * not_clamp_y
        static_collision_n_loss_y = torch.max(static_collision_n_loss1_y, static_collision_n_loss2_y).sum()

        # # # collision
        dynamic_collosion_degree_weight = 1.0 if abs(ego2match_yaw_degrees) > 10.0 else 0.0
        dynamic_collision_advantage_p_weight = 1.0 if (dynamic_collision_advantage < 0.0 and has_dynamic_collision) else 0.0
        dynamic_collision_p_loss1_y = dynamic_collision_advantage_p_weight * (-dynamic_collision_advantage) * (-pseudo_odo_pred_probs_y) * not_clamp_y
        dynamic_collision_p_loss2_y = dynamic_collision_advantage_p_weight * (-dynamic_collision_advantage) * (-pseudo_odo_pred_probs_y) * not_clamp_y
        dynamic_collision_p_loss_y = torch.min(dynamic_collision_p_loss1_y, dynamic_collision_p_loss2_y).sum() * dynamic_collosion_degree_weight

        # collision_n_loss
        dynamic_collision_advantage_n_weight = 1.0 if (dynamic_collision_advantage < 0.0 and has_dynamic_collision) else 0.0
        dynamic_collision_n_loss1_y = dynamic_collision_advantage_n_weight * (dynamic_collision_advantage) * (-neg_odo_pred_probs_y) * not_clamp_y
        dynamic_collision_n_loss2_y = dynamic_collision_advantage_n_weight * (dynamic_collision_advantage) * (-neg_odo_pred_probs_y) * not_clamp_y
        dynamic_collision_n_loss_y = torch.max(dynamic_collision_n_loss1_y, dynamic_collision_n_loss2_y).sum() * dynamic_collosion_degree_weight

        # # # angle
        deviation_angle_p_loss1_y = (-angle_deviation_advantage) * (-pseudo_odo_pred_probs_y) * not_clamp_y
        deviation_angle_p_loss2_y = (-angle_deviation_advantage) * (-pseudo_odo_pred_probs_y) * not_clamp_y
        deviation_angle_p_loss_y = torch.min(deviation_angle_p_loss1_y, deviation_angle_p_loss2_y).sum()

        # angle_n_loss
        deviation_angle_n_loss1_y = angle_deviation_advantage * (-neg_odo_pred_probs_y) * not_clamp_y 
        deviation_angle_n_loss2_y = angle_deviation_advantage * (-neg_odo_pred_probs_y) * not_clamp_y
        deviation_angle_n_loss_y = torch.max(deviation_angle_n_loss1_y, deviation_angle_n_loss2_y).sum()

        # ================================all loss==============================================
        loss_plan_rl_dynamic_coli_x = dynamic_collision_p_loss_x + dynamic_collision_n_loss_x
        loss_plan_rl_dynamic_coli_y = dynamic_collision_p_loss_y + dynamic_collision_n_loss_y
        loss_plan_rl_static_coli_y = static_collision_p_loss_y + static_collision_n_loss_y
        loss_plan_rl_devi_distance = deviation_distance_p_loss_y + deviation_distance_n_loss_y
        loss_plan_rl_devi_angle = deviation_angle_p_loss_y + deviation_angle_n_loss_y
        v_collision_loss = 0.5 * ((new_collision_value - b_collision_return) ** 2).mean()
        
        loss_plan_rl = (
                        loss_plan_rl_dynamic_coli_x 
                        + loss_plan_rl_dynamic_coli_y 
                        + loss_plan_rl_static_coli_y 
                        + loss_plan_rl_devi_distance
                        + loss_plan_rl_devi_angle
                        + pg_loss_x 
                        + pg_loss_y
                        + v_collision_loss
                        )

        if rl_metainfo['collided_this_frame'] or rl_metainfo['game_over']:
            loss_plan_rl = 0. * loss_plan_rl

        return loss_plan_rl 

    def il_loss(self, preds_dicts, ego_fut_gt):

        loss_dict = {}

        outputs_ego_x_action = preds_dicts['outputs_ego_x_action'].squeeze(-1)
        outputs_ego_y_action = preds_dicts['outputs_ego_y_action'].squeeze(-1)

        x_best_anchor_idxs_list = []
        y_best_anchor_idxs_list = []
        for i in range(0, ego_fut_gt.shape[0]):
            ego_x, ego_y = ego_fut_gt[i, 5]
            anchor_xy_05 = self.plan_anchors[:, 5, :]

            x_min, x_max = anchor_xy_05[:, 0].min(), anchor_xy_05[:, 0].max()
            y_min, y_max = anchor_xy_05[:, 1].min(), anchor_xy_05[:, 1].max()

            x_normalized = (anchor_xy_05[:, 0] - x_min) / (x_max - x_min + 1e-6)
            y_normalized = (anchor_xy_05[:, 1] - y_min) / (y_max - y_min + 1e-6)
            ego_x_normalized = (ego_x - x_min) / (x_max - x_min + 1e-6)
            ego_y_normalized = (ego_y - y_min) / (y_max - y_min + 1e-6)

            distances = torch.sqrt(
                (x_normalized - ego_x_normalized) ** 2 +
                (y_normalized - ego_y_normalized) ** 2
            )
            distances[~self.plan_anchors_mask] = 1000
            closest_index = torch.argmin(distances)

            x_best_anchor_idx = (closest_index // self.y_anchor).to(outputs_ego_x_action.device)
            y_best_anchor_idx = (closest_index % self.y_anchor).to(outputs_ego_y_action.device)

            x_best_anchor_idxs_list.append(x_best_anchor_idx[None])
            y_best_anchor_idxs_list.append(y_best_anchor_idx[None])

        x_best_anchor_idxs = torch.cat(x_best_anchor_idxs_list, dim=0)
        y_best_anchor_idxs = torch.cat(y_best_anchor_idxs_list, dim=0)

        loss_discrete_plan_x = self.loss_discrete_plan_x(outputs_ego_x_action , x_best_anchor_idxs)
        loss_discrete_plan_y = self.loss_discrete_plan_y(outputs_ego_y_action , y_best_anchor_idxs)

        loss_dict['loss_discrete_plan_x'] = loss_discrete_plan_x
        loss_dict['loss_discrete_plan_y'] = loss_discrete_plan_y
        print(loss_dict)

if __name__ == "__main__":
    # ===============================
    # Configuration
    # ===============================
    x_anchor = 61
    y_anchor = 61
    fut_ts = 6
    device = "cpu"

    model = RADModel(x_anchor=x_anchor, y_anchor=y_anchor, fut_ts=fut_ts)

    # ===============================
    # 1. Test Imitation Learning (IL) Loss
    # ===============================
    print("===========================1. Test Imitation Learning (IL) Loss===========================")
    batch = 2  # batch size
    # Simulated model outputs (logits over anchors)
    preds_dicts = {
        'outputs_ego_x_action': torch.randn(batch, x_anchor, 1, device=device, requires_grad=True),
        'outputs_ego_y_action': torch.randn(batch, y_anchor, 1, device=device, requires_grad=True)
    }
    # Simulated ground-truth future trajectory, shape [B, T, 2]
    ego_fut_gt = torch.rand(batch, fut_ts, 2) * 10 

    # Compute IL loss
    model.il_loss(preds_dicts, ego_fut_gt)

    # ===============================
    # 2. Test Validation Function
    # ===============================
    print("===========================2. Test Validation Function===========================")
    batch = 1  # batch size
    # Simulated model outputs (logits over anchors)
    preds_dicts = {
        "outputs_ego_x_action": torch.randn(batch, x_anchor, 1, device=device),
        "outputs_ego_y_action": torch.randn(batch, y_anchor, 1, device=device),
    }
    # Run validation
    model.val(preds_dicts)

    # ===============================
    # 3. Test RL Loss Computation
    # ===============================
    print("===========================3. Test RL Loss Computation===========================")
    batch = 1

    # Simulated model outputs (logits over anchors)
    preds_dicts = {
        "rl_value_function": torch.randn(batch, 1, 1, device=device),
        "outputs_ego_x_action": torch.randn(batch, x_anchor, 1, device=device),
        "outputs_ego_y_action": torch.randn(batch, y_anchor, 1, device=device),
    }

    rl_metainfo = {
        # === Advantage signals (used for RL loss weighting) ===
        # Advantage value related to dynamic collisions
        "dynamic_collision_advantage": np.random.randn(1, 1, 1).astype(np.float32),
        # Advantage value related to static collisions 
        "static_collision_advantage": np.random.randn(1, 1, 1).astype(np.float32),
        # Advantage value related to lateral deviation distance from expert trajectory
        "distance_deviation_advantage": np.random.randn(1, 1, 1).astype(np.float32),
        # Advantage value related to heading angle deviation from expert trajectory
        "angle_deviation_advantage": np.random.randn(1, 1, 1).astype(np.float32),
        # === Policy anchor selection ===
        # Selected X-direction anchor index (0 ~ x_anchor-1)
        "selected_anchor_idx_x": 10,
        # Selected Y-direction anchor index (0 ~ y_anchor-1)
        "selected_anchor_idx_y": 20,
        # === Policy outputs (action predictions) ===
        # Predicted distribution over X anchors (probabilities or logits)
        "outputs_ego_x_action_pred": np.random.randn(1, x_anchor).astype(np.float32),
        # Predicted distribution over Y anchors
        "outputs_ego_y_action_pred": np.random.randn(1, y_anchor).astype(np.float32),
        # === Rollout feedback (historical values and deviation measures) ===
        # Previous collision value estimate, used for advantage comparison
        "old_collision_value": np.random.randn(1, 1).astype(np.float32),
        # Lateral deviation distance from the expert trajectory
        "distance_deviation": np.random.randn(1).astype(np.float32),
        # Angular difference between ego action and expert trajectory (in degrees)
        "ego2match_yaw_degrees": np.random.randn(1).astype(np.float32),
        # === Collision-related information ===
        # First detected dynamic collision position (x, y)
        "first_collision_position": np.tile(np.array([1.0, 2.0], dtype=np.float32), (1, 1)),
        # === Flags ===
        # Whether a dynamic collision occurred during the entire rollout (episode-level flag)
        "has_dynamic_collision": True,
        # Whether the ego collided with a collision box at the current frame (frame-level flag)
        "collided_this_frame": False,
        # Whether the rollout has ended (due to collision, success, or timeout)
        "game_over": False,
    }

    loss = model.compute_rl_loss(preds_dicts, rl_metainfo)
    print("RL Loss:", loss.item())