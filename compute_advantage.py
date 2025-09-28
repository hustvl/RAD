import numpy as np

def calculate_longitudinal_jerk(velocity_sequence, time_sequence):
    """
    Calculate longitudinal jerk (rate of change of acceleration).
    
    Args:
        velocity_sequence (list or np.array): Sequence of longitudinal velocities.
        time_sequence (list or np.array): Corresponding time sequence.
    
    Returns:
        jerk_sequence (np.array): Sequence of longitudinal jerk values.
    """
    # Ensure inputs are NumPy arrays
    velocity_sequence = np.array(velocity_sequence)
    time_sequence = np.array(time_sequence)
    
    # Check input lengths
    if len(velocity_sequence) != len(time_sequence):
        raise ValueError("The length of velocity_sequence and time_sequence must match.")
    if len(time_sequence) < 2:
        raise ValueError("The length of time_sequence must be at least 2.")
    
    # Use scipy/numpy gradient to compute acceleration and then jerk
    acceleration_sequence = np.gradient(velocity_sequence, time_sequence)
    jerk_sequence = np.gradient(acceleration_sequence, time_sequence)
    
    return jerk_sequence

def calculate_yaw_jerk(yaw_rate_sequence, time_sequence):
    """
    Calculate the second derivative of yaw angle (Yaw Jerk).
    
    Args:
        yaw_rate_sequence (list or np.array): Sequence of yaw angular velocities.
        time_sequence (list or np.array): Corresponding time sequence.
    
    Returns:
        yaw_jerk_sequence (np.array): Sequence of yaw jerk values (second derivative of yaw angle).
    """
    # Ensure inputs are NumPy arrays
    yaw_rate_sequence = np.array(yaw_rate_sequence)
    time_sequence = np.array(time_sequence)
    
    # Check input lengths
    if len(yaw_rate_sequence) != len(time_sequence):
        raise ValueError("The length of yaw_rate_sequence and time_sequence must match.")
    if len(time_sequence) < 2:
        raise ValueError("The length of time_sequence must be at least 2.")
    
    # Use numpy gradient to compute yaw acceleration and then yaw jerk
    yaw_acc_sequence = np.gradient(yaw_rate_sequence, time_sequence)
    yaw_jerk_sequence = np.gradient(yaw_acc_sequence, time_sequence)
    
    return yaw_jerk_sequence


def compute_advantages_and_metrics(clip_data, d_max):
    """
    clip_data: dict with keys
      'timestamp_list': list of timestamp strings
      'clips_infos': dict {timestamp: info_dict}
    Returns: same return signature as previous Coll_addregular_addstatic_decoupling_valuefunc
    """
    clip_inverse = list(reversed(clip_data['timestamp_list']))
    clips_infos = clip_data['clips_infos']

    dynamic_collision_lastgaelam = 0
    static_collision_lastgaelam = 0
    distance_deviation_lastgaelam = 0
    angle_deviation_lastgaelam = 0
    next_dynamic_collision_value = 0
    next_static_collision_value = 0
    next_distance_deviation_value = 0
    next_angle_deviation_value = 0
    next_is_dynamic_collision = None
    next_is_static_collision = None
    next_distance_deviation = None
    next_angle_deviation = None

    clip_not_gameover_list = []
    velocity_sequence = []
    yaw_rate_sequence = []

    game_over = False
    dynamic_collision_gameover = False
    static_collision_gameover = False
    distance_deviation_gameover = False
    angle_deviation_gameover = False
    front_collision = False
    back_collision = False
    first_collision_position = np.array([0.0, 0.0])
    expert_timestamp = clip_data['timestamp_list'][0]

    for ts in clip_data['timestamp_list']:
        step_info = clips_infos[ts]
        distance_deviation = step_info['distance_deviation']

        step_info['game_over'] = game_over

        if step_info['is_dynamic_collision_box']:
            game_over = True
            dynamic_collision_gameover = not (distance_deviation_gameover or static_collision_gameover or angle_deviation_gameover)

        if distance_deviation > d_max:
            game_over = True
            distance_deviation_gameover = not (dynamic_collision_gameover or static_collision_gameover or angle_deviation_gameover)

        if not step_info['is_dynamic_collision_box'] and step_info['static_collision_score'] > static_collision_score_threshold and distance_deviation > 0.2:
            game_over = True
            static_collision_gameover = not (dynamic_collision_gameover or distance_deviation_gameover or angle_deviation_gameover)

        if abs(step_info['ego2match_yaw_degrees']) > angle_deviation_threshold and distance_deviation > 0.2:
            game_over = True
            angle_deviation_gameover = not (distance_deviation_gameover or static_collision_gameover or dynamic_collision_gameover)

        # first collision position
        if step_info.get("collision_position") is not None and step_info["collision_position"].sum() != 0 and first_collision_position.sum() == 0:
            first_collision_position = step_info["collision_position"]
            if first_collision_position[0] > 0:
                front_collision = True
            else:
                back_collision = True

        clips_infos[ts] = step_info

        if not game_over:
            expert_timestamp = step_info['expert_timestamp']
            clip_not_gameover_list.append(ts)
            velocity_sequence.append(step_info['linear_v'])
            yaw_rate_sequence.append(step_info['yaw_v'])

    for ts in clip_inverse:
        step_info = clips_infos[ts]
        if not step_info['game_over']:
            if next_distance_deviation is None or next_is_dynamic_collision is None or next_angle_deviation is None or next_is_static_collision is None:
                dynamic_collision_reward = 0.0
                static_collision_reward = 0.0
                distance_deviation_reward = 0.0
                angle_deviation_reward = 0.0
                next_distance_deviation = step_info['distance_deviation']
                next_angle_deviation = step_info['ego2match_yaw_degrees']
                step_info['game_over'] = True
            else:
                dynamic_collision_reward = -1.0 if next_is_dynamic_collision else 0.0
                static_collision_reward = -1.0 if next_is_static_collision else 0.0
                distance_deviation_reward = -1.0 if next_distance_deviation > d_max else 0.0
                angle_deviation_reward = -1.0 if (next_distance_deviation > 0.2 and abs(next_angle_deviation) > angle_deviation_threshold) else 0.0

            dynamic_collision_delta = dynamic_collision_reward + collision_gamma * next_dynamic_collision_value - step_info['rl_value_function']
            dynamic_collision_advantage = dynamic_collision_lastgaelam = dynamic_collision_delta + collision_gamma * gae_lambda * dynamic_collision_lastgaelam

            static_collision_delta = static_collision_reward + collision_gamma * next_static_collision_value - 0 * step_info['rl_value_function']
            static_collision_advantage = static_collision_lastgaelam = static_collision_delta + collision_gamma * gae_lambda * static_collision_lastgaelam

            distance_deviation_delta = distance_deviation_reward + distance_deviation_gamma * next_distance_deviation_value - 0 * step_info['rl_value_function']
            distance_deviation_advantage = distance_deviation_lastgaelam = distance_deviation_delta + distance_deviation_gamma * gae_lambda * distance_deviation_lastgaelam

            angle_deviation_delta = angle_deviation_reward + angle_deviation_gamma * next_angle_deviation_value - 0 * step_info['rl_value_function']
            angle_deviation_advantage = angle_deviation_lastgaelam = angle_deviation_delta + angle_deviation_gamma * gae_lambda * angle_deviation_lastgaelam

            next_collision_value = step_info['rl_value_function']  # cache for next step
            next_static_collision_value = 0 * step_info['rl_value_function']  # cache for next step
            next_deviation_value = 0 * step_info['rl_value_function']  # cache for next step
            next_direction_value = 0 * step_info['rl_value_function']  # cache for next step
            next_distance_deviation = step_info['distance_deviation']
            next_angle_deviation = step_info['ego2match_yaw_degrees']
            next_is_dynamic_collision = step_info['is_dynamic_collision_box']
            next_is_static_collision = not step_info['is_dynamic_collision_box'] and step_info['static_collision_score'] > static_collision_score_threshold and step_info['distance_deviation'] > 0.2

            print(f"[{ts}] game_over={step_info['game_over']}, "
                f"dynamic: dyn={dynamic_collision_advantage:.3f}, "
                f"static={static_collision_advantage:.3f}, "
                f"distance={distance_deviation_advantage:.3f}, "
                f"angle={angle_deviation_advantage:.3f}")

        else:
            dynamic_collision_advantage = np.array(0.)
            static_collision_advantage = np.array(0.)
            distance_deviation_advantage = np.array(0.)
            angle_deviation_advantage = np.array(0.)

            print(f"[{ts}] game_over={step_info['game_over']}, "
                f"dynamic: dyn={dynamic_collision_advantage:.3f}, "
                f"static={static_collision_advantage:.3f}, "
                f"distance={distance_deviation_advantage:.3f}, "
                f"angle={angle_deviation_advantage:.3f}")

    velocity_sequence = np.array(velocity_sequence)
    yaw_rate_sequence = np.array(yaw_rate_sequence)
    if len(velocity_sequence) < 2:
        velocity_sequence = np.zeros(2)
        yaw_rate_sequence = np.zeros(2)

    time_sequence = np.arange(0, velocity_sequence.shape[0]) * 0.1
    longitudinal_jerk = calculate_longitudinal_jerk(velocity_sequence, time_sequence)
    yaw_jerk = calculate_yaw_jerk(yaw_rate_sequence, time_sequence)
    longitudinal_jerk_mean = abs(longitudinal_jerk).mean()
    yaw_jerk_mean = abs(yaw_jerk).mean()

    return (
        clip_not_gameover_list,
        dynamic_collision_gameover,
        static_collision_gameover,
        distance_deviation_gameover,
        angle_deviation_gameover,
        first_collision_position,
        expert_timestamp,
        front_collision,
        back_collision,
        longitudinal_jerk_mean,
        yaw_jerk_mean
    )

if __name__ == "__main__":
    """
    Example code for Advantage calculation using simulated data.

    NOTE:
    - This is a demonstration script with placeholder data.
    - Users should replace `clips_infos`  with actual
    collected closed-loop data, aligned properly with timestamps or frames.
    - Hyperparameters provided here are for reference and tuning only.
    """
    # === Hyperparameter definitions ===
    collision_gamma = 0.9
    distance_deviation_gamma = 0.9
    angle_deviation_gamma = 0.9
    gae_lambda = 0.95

    d_max = 2.0
    angle_deviation_threshold = 40
    static_collision_score_threshold = 25000

    # === Simulate example clip data ===
    # This block generates mock trajectory data for demonstration and testing.
    # Each "clip" represents one driving episode containing multiple frames (timestamps).
    # For each frame, various ego-vehicle and environment metrics are stored.
    clips = {}
    num_clips = 3   # Number of mock clips (episodes) to simulate
    num_frames = 6  # Number of frames (timestamps) per clip
    for c in range(num_clips):
        clip_id = f"clip_{c}"
        timestamp_list = [i for i in range(num_frames)]
        clips_infos = {}
        for i, ts in enumerate(timestamp_list):
            clips_infos[ts] = {
                # Ego yaw error relative to the reference trajectory, in degrees.
                'ego2match_yaw_degrees': float(i * 10),
                # Lateral distance deviation from the reference trajectory, in meters.
                'distance_deviation': float(i) * 0.1,
                # Whether a dynamic collision occurred at this frame.
                'is_dynamic_collision_box': (i == 3),
                # Static collision score.
                'static_collision_score': 0,
                # Collision position relative to ego (x, y). Here it's just a placeholder.
                'collision_position': np.array([1.0, 0.0]),
                # Ego vehicle longitudinal velocity (m/s).
                'linear_v': 1.0,
                # Ego vehicle yaw rate (rad/s).
                'yaw_v': 0.1,
                # Expert demonstration timestamp aligned to this frame.
                'expert_timestamp': ts,
                # RL value function output for this frame.
                'rl_value_function': np.random.randn()
            }

        clips[clip_id] = {
            'clip_id': clip_id,
            'timestamp_list': timestamp_list,
            'clips_infos': clips_infos,
        }
    
    # === Compute advantages and return metrics ===
    total_timestamps = 0
    total_not_gameover_timestamps = 0
    clips_EPTR = []
    collision_gameover_list = []
    staric_collision_gameover_list = []
    deviation_gameover_list = []
    direction_gameover_list = []
    success_list = []
    front_collision_list = []
    back_collision_list = []
    longitudinal_jerk_list = []
    yaw_jerk_list = []
    for clip_id, clip_data in clips.items():
        print(f"\n=== Processing clip {clip_id} ===")

        (
            clip_not_gameover_list, 
            dynamic_collision_gameover, 
            static_collision_gameover, 
            distance_deviation_gameover, 
            angle_deviation_gameover, 
            first_collision_position, 
            expert_timestamp, 
            front_collision, 
            back_collision,
            longitudinal_jerk_mean, 
            yaw_jerk_mean
        ) = compute_advantages_and_metrics(
            clip_data,
            d_max
        )

        print("[Trajectory Metrics]")
        print(f"  clip_not_gameover_list   : {clip_not_gameover_list}")
        print(f"  dynamic_collision_gameover       : {dynamic_collision_gameover}")
        print(f"  static_collision_gameover: {static_collision_gameover}")
        print(f"  distance_deviation_gameover       : {distance_deviation_gameover}")
        print(f"  angle_deviation_gameover       : {angle_deviation_gameover}")
        print(f"  first_collision_position : {first_collision_position}")
        print(f"  expert_timestamp         : {expert_timestamp}")
        print(f"  front_collision          : {front_collision}")
        print(f"  back_collision           : {back_collision}")
        print(f"  longitudinal_jerk_mean   : {longitudinal_jerk_mean:.4f}")
        print(f"  yaw_jerk_mean            : {yaw_jerk_mean:.4f}")

        clip_start_timestamp = int(timestamp_list[0])
        clip_end_timestamp = int(timestamp_list[-1])

        clip_EPTR = int(int(expert_timestamp) - int(clip_start_timestamp)) / int(int(clip_end_timestamp) - int(clip_start_timestamp))
        clip_EPTR = max(0, min(1, clip_EPTR))
        clips_EPTR.append(clip_EPTR)

        total_timestamps += len(timestamp_list)
        total_not_gameover_timestamps += len(clip_not_gameover_list)
        if dynamic_collision_gameover and not (distance_deviation_gameover or static_collision_gameover or angle_deviation_gameover):
            collision_gameover_list.append(clip_id)
        elif distance_deviation_gameover and not (dynamic_collision_gameover or static_collision_gameover or angle_deviation_gameover):
            deviation_gameover_list.append(clip_id)
        elif angle_deviation_gameover and not (dynamic_collision_gameover or static_collision_gameover or distance_deviation_gameover):
            direction_gameover_list.append(clip_id)
        elif static_collision_gameover and not (dynamic_collision_gameover or distance_deviation_gameover or angle_deviation_gameover):
            staric_collision_gameover_list.append(clip_id)
        elif not (dynamic_collision_gameover or distance_deviation_gameover or static_collision_gameover or angle_deviation_gameover):
            success_list.append(clip_id)
        else:
            print("error! Multiple gameover conditions detected!")

        if dynamic_collision_gameover and front_collision:
            front_collision_list.append(clip_id)
        elif dynamic_collision_gameover and back_collision:
            back_collision_list.append(clip_id)

        longitudinal_jerk_list.append(longitudinal_jerk_mean)
        yaw_jerk_list.append(yaw_jerk_mean)

    print('success_list:',success_list)
    print('collision_gameover_list:',collision_gameover_list)
    print('staric_collision_gameover_list:',staric_collision_gameover_list)
    print('deviation_gameover_list:',deviation_gameover_list)
    print('direction_gameover_list:',direction_gameover_list)
    print("front_collision_list:",front_collision_list)
    print("back_collision_list:",back_collision_list)

    success_rate = len(success_list) / len(clips)
    collision_rate = len(collision_gameover_list) / len(clips)
    front_collision_rate = len(front_collision_list) / len(clips)
    back_collision_rate = len(back_collision_list) / len(clips)
    static_collision_rate = len(staric_collision_gameover_list) / len(clips)
    deviation_rate = len(deviation_gameover_list) / len(clips)
    direction_rate = len(direction_gameover_list) / len(clips)
    gameover_ratio = total_not_gameover_timestamps / total_timestamps
    average_EPTR = sum(clips_EPTR) / len(clips_EPTR) if len(clips_EPTR) > 0 else 0

    v_jerk = np.array(longitudinal_jerk_list).mean()
    yaw_jerk = np.array(yaw_jerk_list).mean()

    print("SR:"+str(success_rate) + 
          ",\tdynamic_collision_rate:"+str(collision_rate) + 
          ",\tfront_collision_rate:"+str(front_collision_rate) + 
          ",\tback_collision_rate:"+str(back_collision_rate) + 
          ",\tstatic_collision_rate:"+str(static_collision_rate) +
          ",\tdeviation_rate:"+str(deviation_rate) +
          ",\tdirection_rate:"+str(direction_rate) +
          ",\tSDTR:"+str(gameover_ratio) + 
          ",\tEPTR:"+str(average_EPTR) +
          ",\tv_jerk:"+str(v_jerk) + 
          ",\tyaw_jerk:"+str(yaw_jerk)
          )