#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sim_eval_dexora.py

在 dexmimicgen 仿真环境里评估 Dexora 策略，进程内直接调用 DexoraPolicy.get_action，
不走 ZMQ server/client（sim 阶段没有真实的实时性冲突，直接调用更简单也更少出错点）。

支持两种可视化方式：
  1. --render          本地窗口实时渲染（需要有显示器 / X server）
  2. 不加 --render     离屏渲染，把 --viz_camera 指定的机位存成 mp4 视频（默认方式，headless机器也能用）

用法示例：
    # 第一步：先看看这个 env 的 obs 里到底有哪些 key / shape，
    # 用来核对/修改下面 STATE_KEYS 和 CAMERA_KEY_MAP 是否符合你 Step1 定的 M 维语义表
    python sim_eval_dexora.py --env TwoArmCanSortRandom --inspect_obs

    # 跑评估，把每个 episode 存成视频
    python sim_eval_dexora.py \
        --env TwoArmCoffee \
        --model_path /path/to/dexora_ckpt_dir \
        --model_config_path configs/base_400m.yaml \
        --stats_file /path/to/lerobot_stats/dataset_statistics.json \
        --normalize_mode min_max \
        --instruction "pick up the coffee pod and place it in the machine" \
        --n_rollouts 5 --horizon 400 \
        --video_dir ./eval_videos

    # 有显示器的机器上想直接看仿真窗口
    python sim_eval_dexora.py --env TwoArmCoffee --model_path ... --render

export DEXORA_LEROBOT_ROOT=/home/ubuntu/myh/expirement/Dexora/lerobot_data/two_arm_can_sort_random
export DEXORA_STATS=/home/ubuntu/myh/expirement/Dexora/lerobot_data/new_lerobot_stats/dataset_statistics.json
export DEXORA_T5=/home/ubuntu/myh/expirement/Dexora/google/t5-v1_1-small
export DEXORA_SIGLIP=/home/ubuntu/myh/expirement/Dexora/google/siglip-so400m-patch14-384

python sim_eval_dexora.py --env TwoArmCanSortRandom --model_path /home/ubuntu/myh/expirement/Dexora/checkpoints/dexora-400m-pretrain/     --model_config_path /home/ubuntu/myh/expirement/Dexora/configs/base_400m.yaml     --stats_file /home/ubuntu/myh/expirement/Dexora/lerobot_data/new_lerobot_stats/dataset_statistics.json     --camera_height 84 --camera_width 84 --instruction "Use both hands to move the blue can to its sorting bin." --render

python sim_eval_dexora.py --env TwoArmCanSortRandom --model_path /home/ubuntu/myh/expirement/Dexora/checkpoints/dexora-400m-posttrain/     --model_config_path /home/ubuntu/myh/expirement/Dexora/configs/base_400m.yaml     --stats_file /home/ubuntu/myh/expirement/Dexora/lerobot_data/new_lerobot_stats/dataset_statistics.json     --camera_height 84 --camera_width 84 --instruction "Use both hands to move the blue can to its sorting bin." --render

    依赖：robosuite, dexmimicgen, imageio, numpy, 以及你自己的 dexora_policy.py（需要在 PYTHONPATH 里能 import 到）。
"""

import cv2
import argparse
import json
import os
import time

import torch
import numpy as np
import imageio
import robosuite
from robosuite import load_composite_controller_config
from robosuite.utils.transform_utils import quat2axisangle

import dexmimicgen  # noqa: F401  必须 import 才能把自定义环境注册到 robosuite 里

import sys
sys.path.append('/home/ubuntu/myh/expirement')
from Dexora.deploy.dexora_policy import DexoraPolicy, DexoraPolicyConfig  # noqa: E402


# =============================================================================
# 下面两张表是本脚本里唯一需要你根据实际情况核对/修改的地方。
# 先用 --inspect_obs 跑一次，把打印出来的 obs keys 对照你 Step1 定的 M 维语义表填进去。
# =============================================================================

# env_name -> robots 参数，抄自 dexmimicgen_demo_random_action.py 的 ENV_ROBOTS
ENV_ROBOTS = {
    "TwoArmThreading": ["Panda", "Panda"],
    "TwoArmThreePieceAssembly": ["Panda", "Panda"],
    "TwoArmTransport": ["Panda", "Panda"],
    "TwoArmLiftTray": ["PandaDexRH", "PandaDexLH"],
    "TwoArmBoxCleanup": ["PandaDexRH", "PandaDexLH"],
    "TwoArmDrawerCleanup": ["PandaDexRH", "PandaDexLH"],
    "TwoArmCoffee": ["GR1FixedLowerBody"],
    "TwoArmPouring": ["GR1FixedLowerBody"],
    "TwoArmCanSortRandom": ["GR1ArmsOnly"],
}

# robosuite 里实际渲染出来的相机名 -> DexoraPolicy 期望的 DEXORA_CAMERA_ORDER key
# （cam_head / cam_left_wrist / cam_third_view / cam_right_wrist，见 dexora_policy.py）
# 根据 --inspect_obs 在 TwoArmCanSortRandom(GR1ArmsOnly) 上的真实输出核对过：
#   frontview_image / robot0_eye_in_left_hand_image / robot0_eye_in_right_hand_image
# 训练数据里本来就没有真实的 top/head 机位（见之前的记录），所以这里不填 cam_head，
# DexoraPolicy._encode_images 会自动用 SigLIP 均值色把它填成"永久 mask"的占位机位，
# 和你训练时的 zero-pad 处理保持一致——不要在这里塞一个假的头部相机凑数。
CAMERA_KEY_MAP = {
    "frontview": "cam_third_view",
    "robot0_eye_in_left_hand": "cam_left_wrist",
    "robot0_eye_in_right_hand": "cam_right_wrist",
}

# 手部 6->11 的正向映射（来自 fourier_hands.py 源码的 indices 数组）：
# 训练/action 侧是 6 维语义动作，通过这个数组展开成 11 维实际关节指令，
# 其中好几个 11 维关节共享同一个 6 维源（耦合/mimic 关节）。
HAND_INDICES = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5])

# state 是反过来的：从 obs 里 11 维实际 qpos 近似还原成 6 维。
# 对每个 6 维 slot，取它在 HAND_INDICES 里第一次出现的位置，用那一维原始 qpos 代表整组耦合关节
# （因为是 mimic 关节，同一 slot 对应的几个 11 维值理论上应该接近，取第一个即可，是近似值不是精确逆映射）。
_HAND_SLOT_TO_RAW_IDX = [int(np.where(HAND_INDICES == slot)[0][0]) for slot in range(6)]
RIGHT_HAND_SLOT_INDICES = _HAND_SLOT_TO_RAW_IDX
LEFT_HAND_SLOT_INDICES = _HAND_SLOT_TO_RAW_IDX


# 与 fourier_hands.py 的 indices = [0,0,1,1,2,2,3,3,4,4,5] 对应的分组
HAND_MIMIC_GROUPS = [
    [0, 1],   # slot 0
    [2, 3],   # slot 1
    [4, 5],   # slot 2
    [6, 7],   # slot 3
    [8, 9],   # slot 4
    [10],     # slot 5（无耦合关节，单独一个）
]

def hand_qpos_to_6dim(raw_qpos_11, mimic_groups=HAND_MIMIC_GROUPS):
    """把 11 维原始 fourier hand qpos，按 mimic 分组取均值，得到 6 维近似 state。
    与 dexmimicgen_to_lerobot.py 中 out[a_idx] = qpos11[positions].mean() 保持一致。
    """
    raw_qpos_11 = np.asarray(raw_qpos_11).reshape(-1)
    return np.array([raw_qpos_11[g].mean() for g in mimic_groups])


# =============================================================================
# 归一化 / 反归一化 —— 必须和 lerobot_vla_dataset.py 的 _normalize_data 完全一致，
# 否则模型看到的 state 分布、model 输出的 action 分布都和训练时对不上。
#
# 训练时 state/action 都是先转换成 24 维物理量（米/弧度），再用 dataset_statistics.json
# 里的统计量归一化后才喂进网络的；网络学到的也是"归一化空间"里的映射。
# 之前脚本漏了这一步，直接把网络的原始输出当成真实 EEF 坐标喂给 env.step()，
# 这正是你现在看到"model output"比"env实际EEF"大几百到上千倍的原因——
# 网络输出其实还停留在归一化空间，没有反归一化回物理单位。
# =============================================================================

def load_stats(stats_file):
    with open(stats_file, "r") as f:
        return json.load(f)


def normalize(data, stats_entry, mode):
    """物理量 -> 归一化空间，喂给模型前对 state 用。"""
    data = np.asarray(data, dtype=np.float64)
    if mode == "mean_std":
        mean = np.array(stats_entry["mean"])
        std = np.array(stats_entry["std"])
        std = np.where(std == 0, 1, std)
        out = (data - mean) / std
    elif mode == "min_max":
        min_val = np.array(stats_entry["percentile_1"])
        max_val = np.array(stats_entry["percentile_99"])
        rng = max_val - min_val
        rng = np.where(rng == 0, 1, rng)
        out = (data - min_val) / rng
    else:
        raise ValueError(f"未知 normalize_mode: {mode}")
    return out.astype(np.float32)


def denormalize(data, stats_entry, mode):
    """归一化空间 -> 物理量，模型输出的 action 要过这一步才能喂给 env.step()。"""
    data = np.asarray(data, dtype=np.float64)
    if mode == "mean_std":
        mean = np.array(stats_entry["mean"])
        std = np.array(stats_entry["std"])
        out = data * std + mean
    elif mode == "min_max":
        min_val = np.array(stats_entry["percentile_1"])
        max_val = np.array(stats_entry["percentile_99"])
        rng = max_val - min_val
        out = data * rng + min_val
    else:
        raise ValueError(f"未知 normalize_mode: {mode}")
    return out.astype(np.float32)


# =============================================================================
# canonical(模型/数据集列序) -> env 原生 action_spec 顺序的重排。
#
# dexmimicgen_to_lerobot.py 里 STATE_ACTION_NAMES / build_action_vector 输出的是：
#   0:6 右臂 | 6:12 左臂 | 12:18 右手 | 18:24 左手      （canonical，模型学的就是这个顺序）
# 而它的 ACTION_LAYOUT（从 hdf5 原始 action 切片、也就是 env.step() 真正吃的顺序）是：
#   0:6 右臂 | 6:12 右手 | 12:18 左臂 | 18:24 左手      （env 原生顺序）
# 两者不一样！之前的脚本直接把模型输出（canonical 顺序）喂给 env.step()，
# 相当于把"左臂目标"当成"右手目标"喂进去、把"右手目标"当成"左臂目标"喂进去——
# 这也是机械臂/手乱动的另一个直接原因，和上面缺反归一化是两个独立的 bug，要一起改。
#
# ⚠️ 如果你的 ACTION_LAYOUT 和 dexmimicgen_to_lerobot.py 里的不一样（比如后来跑
# --inspect 改过），这个函数要跟着改。
# =============================================================================

def canonical_action_to_env(action_24):
    action_24 = np.asarray(action_24).reshape(-1)
    right_arm = action_24[0:6]
    left_arm = action_24[6:12]
    right_hand = action_24[12:18]
    left_hand = action_24[18:24]
    return np.concatenate([right_arm, right_hand, left_arm, left_hand]).astype(np.float32)


# =============================================================================
# env / obs 相关工具函数
# =============================================================================

def make_env(env_name, camera_names, camera_height=384, camera_width=384, has_renderer=False):
    if env_name not in ENV_ROBOTS:
        raise ValueError(f"未知 env: {env_name}，请检查 ENV_ROBOTS 里有没有这个 key")
    robots = ENV_ROBOTS[env_name]
    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=load_composite_controller_config(robot=robots[0]),
        has_renderer=has_renderer,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=camera_names,
        camera_heights=camera_height,
        camera_widths=camera_width,
        control_freq=20,
    )
    return robosuite.make(**env_kwargs)


def inspect_obs(env_name, camera_names, camera_height=384, camera_width=384):
    """只 reset 一次，把 obs 里所有 key/shape 打印出来，不涉及 policy。"""
    env = make_env(env_name, camera_names, camera_height, camera_width, has_renderer=False)
    obs = env.reset()
    print(f"=== obs keys for env={env_name} (robots={ENV_ROBOTS[env_name]}) ===")
    for k in sorted(obs.keys()):
        v = obs[k]
        shape = getattr(v, "shape", None)
        dtype = getattr(v, "dtype", type(v))
        print(f"  {k:35s} shape={shape} dtype={dtype}")
    env.close()


def build_state(obs):
    """
    按 Step1 的 24 维 (M) 语义表拼 state：
        0:6   右臂 eef pose (xyz(3) + axis-angle(3))
        6:12  左臂 eef pose (xyz(3) + axis-angle(3))
        12:18 右手 6 维（从 11 维原始 qpos 按 slot index 取出来）
        18:24 左手 6 维
    obs 里的 eef 姿态是四元数（*_eef_quat），要转成 axis-angle；
    手部 obs 是 11 维原始关节角，要按 RIGHT_HAND_SLOT_INDICES / LEFT_HAND_SLOT_INDICES 取成 6 维。
    """
    required = [
        "robot0_right_eef_pos", "robot0_right_eef_quat",
        "robot0_left_eef_pos", "robot0_left_eef_quat",
        "robot0_right_gripper_qpos", "robot0_left_gripper_qpos",
    ]
    missing = [k for k in required if k not in obs]
    if missing:
        raise KeyError(
            f"obs 里缺少 key: {missing}。先跑 `--inspect_obs` 核对这个 env 实际的 obs keys。"
        )

    right_pos = np.asarray(obs["robot0_right_eef_pos"]).reshape(-1)
    right_aa = quat2axisangle(np.asarray(obs["robot0_right_eef_quat"]).reshape(-1))
    left_pos = np.asarray(obs["robot0_left_eef_pos"]).reshape(-1)
    left_aa = quat2axisangle(np.asarray(obs["robot0_left_eef_quat"]).reshape(-1))

    right_hand6 = hand_qpos_to_6dim(obs["robot0_right_gripper_qpos"], RIGHT_HAND_SLOT_INDICES)
    left_hand6 = hand_qpos_to_6dim(obs["robot0_left_gripper_qpos"], LEFT_HAND_SLOT_INDICES)

    state = np.concatenate(
        [right_pos, right_aa, left_pos, left_aa, right_hand6, left_hand6], axis=0
    )
    return state.astype(np.float32)


# 直接对齐 siglip-so400m-patch14-384 的输入尺寸（384x384），避免训练时先resize到256
# 再被 SigLIP processor 二次resize到384，两次插值会比一次插值更糊。
IMAGE_SIZE = (384, 384)  # (H, W)

def build_images(obs, camera_key_map):
    images = {}
    for robosuite_cam, dexora_cam in camera_key_map.items():
        img_key = f"{robosuite_cam}_image"
        if img_key in obs:
            # robosuite 默认图像是上下翻转的（OpenGL 惯例），送进视觉编码器 / 存视频前翻回来
            img = obs[img_key][::-1]
            if img.shape[:2] != IMAGE_SIZE:
                img = cv2.resize(img, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_CUBIC)
            images[dexora_cam] = img
    return images


# =============================================================================
# action chunk 的队列消费逻辑：一次 get_action 拿 chunk_size 步，逐步喂给 env，
# 消费完再重新 query 一次策略。
# =============================================================================

class ChunkActionQueue:
    def __init__(self):
        self._queue = []

    def empty(self):
        return len(self._queue) == 0

    def push_chunk(self, chunk):
        self._queue = list(np.asarray(chunk))

    def pop(self):
        return self._queue.pop(0)


def rollout_episode(
    env,
    policy,
    instruction,
    horizon,
    camera_key_map,
    stats,
    normalize_mode,
    ctrl_freq=20.0,
    viz_camera="agentview",
    video_writer=None,
    live_render=False,
):
    obs = env.reset()
    if live_render:
        env.render()

    action_queue = ChunkActionQueue()
    success = False
    t = 0
    for t in range(horizon):
        if action_queue.empty():
            state_raw = build_state(obs)
            state_norm = normalize(state_raw, stats["state"], normalize_mode)
            images = build_images(obs, camera_key_map)
            policy_obs = {
                "state": state_norm,
                "images": images,
                "instruction": instruction,
                "ctrl_freq": ctrl_freq,
            }
            action_chunk = policy.get_action(policy_obs)  # [chunk_size, M]，模型输出仍是归一化+canonical顺序
            print("raw normalized output:", action_chunk[0])
            action_chunk = denormalize(action_chunk, stats["action"], normalize_mode)
            action_chunk = np.stack([canonical_action_to_env(a) for a in action_chunk], axis=0)
            action_queue.push_chunk(action_chunk)

        action = action_queue.pop()
        right_eef_before_world = np.asarray(obs["robot0_right_eef_pos"]).reshape(-1)
        right_eef_before_base = np.asarray(
            obs.get("robot0_base_to_right_eef_pos", obs["robot0_right_eef_pos"])
        ).reshape(-1)

        # 打印调试日志，验证 Z 轴数值是否恢复到了 1.1 米左右
        if t % 20 == 0:
            print("right arm", action[0:6])
            print("left arm ", action[6:12])
            print("right hand", action[12:18])
            print("left hand", action[18:24])
            print(f"[Step {t}] Model Output Right EEF Target:", action[0:3])
            print(f"[Step {t}] Env Actual Right EEF Before Step (world):", right_eef_before_world)
            print(f"[Step {t}] Env Actual Right EEF Before Step (base-relative):", right_eef_before_base)

        obs, reward, done, info = env.step(action)

        if t % 20 == 0:
            right_eef_after_world = np.asarray(obs["robot0_right_eef_pos"]).reshape(-1)
            right_eef_after_base = np.asarray(
                obs.get("robot0_base_to_right_eef_pos", obs["robot0_right_eef_pos"])
            ).reshape(-1)
            print(f"[Step {t}] Env Actual Right EEF After Step (world):", right_eef_after_world)
            print(f"[Step {t}] Env Actual Right EEF After Step (base-relative):", right_eef_after_base)
            print(f"[Step {t}] Right EEF Delta After Step (world-target):", right_eef_after_world - action[0:3])
            print(f"[Step {t}] Right EEF Delta After Step (base-target):", right_eef_after_base - action[0:3])

        if live_render:
            env.render()
        if video_writer is not None:
            frame_key = f"{viz_camera}_image"
            if frame_key in obs:
                video_writer.append_data(obs[frame_key][::-1])

        if hasattr(env, "_check_success") and env._check_success():
            success = True
            break
        if done:
            break

    return success, t + 1


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="TwoArmCoffee")
    parser.add_argument("--model_path", type=str, default=None, help="Dexora checkpoint 目录/文件路径")
    parser.add_argument("--model_config_path", type=str, default="configs/base_400m.yaml")
    parser.add_argument("--instruction", type=str, default="", help="固定语言指令，对齐训练时的某一条 phrasing")
    parser.add_argument("--n_rollouts", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--state_dim", type=int, default=24, help="对应 Step1/2 里确定的 M 维")
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--video_dir", type=str, default="./eval_videos")
    parser.add_argument("--viz_camera", type=str, default="agentview", help="存视频用哪个机位")
    parser.add_argument("--camera_height", type=int, default=384)
    parser.add_argument("--camera_width", type=int, default=384)
    parser.add_argument("--render", action="store_true", help="本地窗口实时渲染，和存视频二选一")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stats_file", type=str, default=None,
        help="dataset_statistics.json 路径（lerobot_vla_dataset.py --stat 生成的），必须提供才能正确归一化/反归一化",
    )
    parser.add_argument(
        "--normalize_mode", type=str, default="min_max", choices=["min_max", "mean_std"],
        help="要和训练时 LeRobotVLADataset(normalize_mode=...) 用的模式一致",
    )
    parser.add_argument(
        "--inspect_obs",
        action="store_true",
        help="只打印一次 obs keys/shape 就退出，不加载 policy，用来配置 STATE_KEYS/CAMERA_KEY_MAP",
    )
    args = parser.parse_args()

    camera_names = sorted(set(CAMERA_KEY_MAP.keys()) | {args.viz_camera})

    if args.inspect_obs:
        inspect_obs(args.env, camera_names, args.camera_height, args.camera_width)
        return

    if args.model_path is None:
        raise ValueError("--model_path 必须提供（除非只是 --inspect_obs）")
    if args.stats_file is None:
        raise ValueError(
            "--stats_file 必须提供（dataset_statistics.json），否则 state/action 没法正确归一化/反归一化，"
            "模型输出会停留在训练时的归一化空间，直接喂给 env 会得到离谱的大数值。"
        )

    stats = load_stats(args.stats_file)

    np.random.seed(args.seed)

    cfg = DexoraPolicyConfig(
        model_config_path=args.model_config_path,
        state_dim=args.state_dim,
        chunk_size=args.chunk_size,
        # dtype=torch.float32,
    )
    policy = DexoraPolicy(model_path=args.model_path, cfg=cfg)

    env = make_env(
        args.env, camera_names, args.camera_height, args.camera_width, has_renderer=args.render
    )

    if not args.render:
        os.makedirs(args.video_dir, exist_ok=True)

    n_success = 0
    for ep in range(args.n_rollouts):
        writer = None
        video_path = None
        if not args.render:
            video_path = os.path.join(args.video_dir, f"{args.env}_ep{ep}.mp4")
            writer = imageio.get_writer(video_path, fps=20)

        t0 = time.time()
        success, n_steps = rollout_episode(
            env,
            policy,
            args.instruction,
            args.horizon,
            CAMERA_KEY_MAP,
            stats,
            args.normalize_mode,
            viz_camera=args.viz_camera,
            video_writer=writer,
            live_render=args.render,
        )
        if writer is not None:
            writer.close()
        dt = time.time() - t0
        n_success += int(success)

        msg = f"[ep {ep}] success={success} steps={n_steps} time={dt:.1f}s"
        if video_path is not None:
            msg += f" video={video_path}"
        print(msg)

    print(f"=== success rate: {n_success}/{args.n_rollouts} ===")
    env.close()


if __name__ == "__main__":
    main()
