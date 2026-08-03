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
    python sim_eval_dexora.py --env TwoArmCoffee --inspect_obs

    # 跑评估，把每个 episode 存成视频
    python sim_eval_dexora.py \
        --env TwoArmCoffee \
        --model_path /path/to/dexora_ckpt_dir \
        --model_config_path configs/base_400m.yaml \
        --instruction "pick up the coffee pod and place it in the machine" \
        --n_rollouts 5 --horizon 400 \
        --video_dir ./eval_videos

    # 有显示器的机器上想直接看仿真窗口
    python sim_eval_dexora.py --env TwoArmCoffee --model_path ... --render

依赖：robosuite, dexmimicgen, imageio, numpy, 以及你自己的 dexora_policy.py（需要在 PYTHONPATH 里能 import 到）。
"""

import argparse
import os
import time

import numpy as np
import imageio
import robosuite
from robosuite import load_composite_controller_config

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
# 用 --inspect_obs 打印出来的 "*_image" key 核对左边这一列，右边按你训练数据用的机位对应关系改。
CAMERA_KEY_MAP = {
    "agentview": "cam_third_view",
    "robot0_eye_in_left_hand": "cam_left_wrist",   # 对应 DEXORA 的左手腕视角
    "robot0_eye_in_right_hand": "cam_right_wrist", # 对应 DEXORA 的右手腕视角
    "frontview": "cam_head",                       # 将头部视角映射至 frontview (或根据你模型的训练数据机位映射)
}

# 按 Step1 定的 24 维 (M) 语义表，从 obs dict 里按顺序取 key 拼成 state：
#   0:6   右臂 eef pose (xyz + axis-angle)
#   6:12  左臂 eef pose (xyz + axis-angle)
#   12:18 右手 6 维（对应 fourier_hands.py 的 6 维 slot-mapped action）
#   18:24 左手 6 维
# 下面这份是占位，key 名字未必和你实际 env 的 obs 完全一致，务必先用 --inspect_obs 核对一遍。
STATE_KEYS = [
    ("robot0_right_eef_pos", 3),
    ("robot0_right_eef_axisangle", 3),
    ("robot0_left_eef_pos", 3),
    ("robot0_left_eef_axisangle", 3),
    ("robot0_right_gripper_qpos", 6),
    ("robot0_left_gripper_qpos", 6),
]


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


def build_state(obs, state_keys):
    parts = []
    for key, dim in state_keys:
        if key not in obs:
            raise KeyError(
                f"obs 里没有 key='{key}'。先跑 `--inspect_obs` 打印实际的 obs keys，"
                f"再按 Step1 的 M 维语义表修改脚本顶部的 STATE_KEYS。"
            )
        v = np.asarray(obs[key]).reshape(-1)
        if v.shape[0] != dim:
            raise ValueError(
                f"key='{key}' 期望 {dim} 维，实际拿到 {v.shape[0]} 维，检查 STATE_KEYS 配置是否和这个 env 匹配"
            )
        parts.append(v)
    return np.concatenate(parts, axis=0)


def build_images(obs, camera_key_map):
    images = {}
    for robosuite_cam, dexora_cam in camera_key_map.items():
        img_key = f"{robosuite_cam}_image"
        if img_key in obs:
            # robosuite 默认图像是上下翻转的（OpenGL 惯例），送进视觉编码器 / 存视频前翻回来
            images[dexora_cam] = obs[img_key][::-1]
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
            state = build_state(obs, STATE_KEYS)
            images = build_images(obs, camera_key_map)
            policy_obs = {
                "state": state,
                "images": images,
                "instruction": instruction,
                "ctrl_freq": ctrl_freq,
            }
            action_chunk = policy.get_action(policy_obs)  # [chunk_size, M]
            action_queue.push_chunk(action_chunk)

        action = action_queue.pop()
        obs, reward, done, info = env.step(action)

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

    np.random.seed(args.seed)

    cfg = DexoraPolicyConfig(
        model_config_path=args.model_config_path,
        state_dim=args.state_dim,
        chunk_size=args.chunk_size,
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
