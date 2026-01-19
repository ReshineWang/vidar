import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from .ar import AR


def encode_obs(observation):  # Post-Process Observation
    obs = observation

    head_rgbs = np.array(obs['observation']["head_camera"]["rgb"])
    left_rgbs = np.array(obs['observation']["left_camera"]["rgb"])
    right_rgbs = np.array(obs['observation']["right_camera"]["rgb"])

    # rgb to bgr
    head_rgbs = head_rgbs[..., ::-1]
    left_rgbs = left_rgbs[..., ::-1]
    right_rgbs = right_rgbs[..., ::-1]
        
    # This logic assumes a single frame observation, not a sequence.
    # The original logic seemed to expect a sequence (len(head_rgbs)).
    # We'll process a single frame observation dict.
    head_img = head_rgbs
    left_img = left_rgbs
    right_img = right_rgbs
    
    h, w, _ = head_img.shape
    new_h, new_w = h // 2, w // 2
    
    left_resized = cv2.resize(left_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(right_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    bottom_row = np.concatenate([left_resized, right_resized], axis=1)
    
    assert bottom_row.shape[1] == w

    final_h = h + new_h
    combined_img = np.zeros((final_h, w, 3), dtype=head_img.dtype)
    
    combined_img[:h, :w] = head_img
    combined_img[h:, :w] = bottom_row
    combined_img = combined_img[:, :, ::-1].copy()
    return combined_img


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    """Initializes and returns the Vidar policy model."""
    model = AR(usr_args=usr_args)
    return model


def eval(TASK_ENV, model, observation):
    """
    The AR model now handles the full observation-to-action pipeline.
    """
    TASK_ENV.step_lim = model.max_steps
    obs = encode_obs(observation)

    # Set instruction and update observation for the model
    model.set_episode_id(TASK_ENV.ep_num)
    if model.task_config.startswith("demo"):
        instruction = TASK_ENV.instruction
        model.set_demo_instruction(instruction)
    else:
        instruction = TASK_ENV.full_instruction
        model.set_instruction(instruction)
    model.update_obs(obs)
    print(f"Instruction: {model.prompt}")

    # AR model now directly returns a sequence of actions
    actions = model.get_actions()
    action_idx = 0

    # --- Action Execution Loop ---
    while action_idx < len(actions):
        action = actions[action_idx]
        TASK_ENV.take_action(action, action_type='qpos')
        if TASK_ENV.eval_success:
            break
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)
        action_idx += 1
        if action_idx == len(actions):
            actions += model.get_actions()
    
    if model.usr_args.get("eval_action_log", False):
        try:
            save_dir = TASK_ENV.eval_video_path
            if save_dir:
                actions_np = np.array(actions)
                fig, axs = plt.subplots(14, 1, figsize=(10, 20))
                for i in range(min(14, actions_np.shape[1])):
                    axs[i].plot(actions_np[:, i], color='blue')
                    axs[i].set_title(f"Dim {i}")
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"episode{TASK_ENV.test_num}_action.png"))
                plt.close()
                np.save(os.path.join(save_dir, f"episode{TASK_ENV.test_num}_action.npy"), actions_np)
        except Exception as e:
            print(f"Failed to log actions: {e}")

    model.save_videos()


def reset_model(model):  
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.reset()
