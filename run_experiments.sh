#!/bin/bash

# 1. 실험 설정 리스트 (형식 - "알고리즘:체크포인트경로:디렉토리접미사")
# MPC는 체크포인트가 없으므로 공백으로 비워둡니다.
EXPERIMENTS=(
    "mpc::default"
    "ppo:/home/mjkim/Underwater-Actor-Critic-Model-Predictive-Control1/wandb/PPO-hybrid-disturbance/files/checkpoint_final.pt:lr_1e-4"
    #"ppo:/home/mjkim/.../ppo_v2/checkpoint_final.pt:lr_5e-5"
    "acmpc:/home/mjkim/ACMPC-random-disturbances/files/checkpoint_final.pt:w_reward_1.0"
    #"acmpc:/home/mjkim/.../acmpc_v2/checkpoint_5000000.pt:w_reward_2.0"
)

BASE_OUT_DIR="/home/mjkim/Underwater-Actor-Critic-Model-Predictive-Control1/outputs/$(date +%Y-%m-%d_%H-%M-%S)_batch"

for EXP in "${EXPERIMENTS[@]}"; do
    # 콜론(:)을 기준으로 데이터 분리
    IFS=":" read -r ALGO CKPT SUFFIX <<< "$EXP"
    
    # 출력 디렉토리 이름 결정 (예: ppo_lr_1e-4)
    FOLDER_NAME="${ALGO}_${SUFFIX}"
    OUT_DIR="$BASE_OUT_DIR/$FOLDER_NAME"
    mkdir -p "$OUT_DIR"

    echo "=========================================================="
    echo "Running Experiment: $FOLDER_NAME"
    echo "Algorithm: $ALGO | Suffix: $SUFFIX"
    [ ! -z "$CKPT" ] && echo "Checkpoint: $CKPT"
    echo "=========================================================="

    # --- 공통 파라미터 설정 ---
    COMMON_ARGS="headless=true env.num_envs=1 mode=evaluate +eval.steps=4000"
    DISTURB_ARGS="task.disturbances.evaluate.payload.enable_payload=true task.disturbances.evaluate.payload.mass='[0.1,0.1]' task.disturbances.evaluate.payload.z='[0.0,0.0]'"

    # --- Step 1: Trajectory Plotting ---
    echo "[Step 1] Plotting Trajectory..."
    if [ "$ALGO" == "mpc" ]; then
        ~/isaac410/python.sh scripts/test_mpc_withCam.py task=OrbitCylinder_MPC $COMMON_ARGS task.use_pypose_mpc=false +camera.head_offset='[0.4,0,0.15]' mode=evaluate
        mv ./trajectory.npz ./trajectory.png ./trajectory_energy.png ./trajectory_energy_polar.png $OUT_DIR
    elif [ "$ALGO" == "ppo" ]; then
        ~/isaac410/python.sh scripts/evaluate.py task=OrbitCylinder_MPC_PPO algo=ppo $COMMON_ARGS +eval.ckpt="$CKPT" mode=evaluate
        mv ./trajectory.npz ./trajectory.png ./trajectory_energy.png ./trajectory_energy_polar.png $OUT_DIR
    elif [ "$ALGO" == "acmpc" ]; then
        ~/isaac410/python.sh scripts/evaluate_pypose_mpc_ppo.py task=OrbitCylinder_MPC algo=ppo_pypose_cylinder_mpc_werr_wu_tv \
        task.reward_mode=orbit_ppo task.orbit_target_mode=auto task.use_internal_mpc=false \
        task.include_cylinder_rel_in_obs=false $COMMON_ARGS +eval.ckpt="$CKPT" mode=evaluate
        mv ./trajectory.npz ./trajectory.png ./trajectory_energy.png ./trajectory_energy_polar.png $OUT_DIR
    fi

    # 시각화 후 처리 (최신 생성된 npz 파일 이동 및 이미지 생성)
    #LATEST_NPZ=$(find outputs -name "trajectory.npz" | xargs ls -t | head -n 1)
    ~/isaac410/python.sh scripts/visualize_trajectory.py $OUT_DIR/trajectory.npz --traj-color-mode termination --color-key '' --out "$OUT_DIR/traj_${SUFFIX}.png"
    #cp "$LATEST_NPZ" "$OUT_DIR/trajectory.npz"

    # --- Step 2: Current Sweep Energy ---
    echo "[Step 2] Current Sweep..."
    SWEEP_OPTS="+eval.current_speeds='[0.0,0.05,0.10,0.15,0.20]' +eval.current_dir='[1,0,0]' +eval.episodes_per_speed=5 env.max_episode_length=400"
    
    if [ "$ALGO" == "mpc" ]; then
        ~/isaac410/python.sh scripts/evaluate_current_sweep_energy.py task=OrbitCylinder_MPC algo=ppo headless=true env.num_envs=1 mode=evaluate +eval.steps=4000 task.disturbances.evaluate.payload.enable_payload=true task.disturbances.evaluate.payload.mass='[0.1,0.1]' task.disturbances.evaluate.payload.z='[0.0,0.0]' +eval.current_speeds='[0.0,0.05,0.10,0.15,0.20]' +eval.current_dir='[1,0,0]' +eval.episodes_per_speed=5 env.max_episode_length=400 +eval.controller=internal_mpc task.use_pypose_mpc=false
        mv ./current_vs_energy_mpc.npz ./current_vs_energy_mpc.png $OUT_DIR
    elif [ "$ALGO" == "ppo" ]; then
        ~/isaac410/python.sh scripts/evaluate_current_sweep_energy.py task=OrbitCylinder_MPC_PPO algo=ppo headless=true env.num_envs=1 mode=evaluate +eval.steps=4000 task.disturbances.evaluate.payload.enable_payload=true task.disturbances.evaluate.payload.mass='[0.1,0.1]' task.disturbances.evaluate.payload.z='[0.0,0.0]' +eval.current_speeds='[0.0,0.05,0.10,0.15,0.20]' +eval.current_dir='[1,0,0]' +eval.episodes_per_speed=5 env.max_episode_length=400 +eval.ckpt="$CKPT"
        mv ./current_vs_energy.npz ./current_vs_energy.png $OUT_DIR
    elif [ "$ALGO" == "acmpc" ]; then
        ~/isaac410/python.sh scripts/evaluate_current_sweep_energy.py task=OrbitCylinder_MPC algo=ppo_pypose_cylinder_mpc_werr_wu_tv headless=true env.num_envs=1 mode=evaluate +eval.steps=4000 task.disturbances.evaluate.payload.enable_payload=true task.disturbances.evaluate.payload.mass='[0.1,0.1]' task.disturbances.evaluate.payload.z='[0.0,0.0]' +eval.current_speeds='[0.0,0.05,0.10,0.15,0.20]' +eval.current_dir='[1,0,0]' +eval.episodes_per_speed=5 env.max_episode_length=400 \
        +eval.controller=policy +eval.ckpt="$CKPT" task.reward_mode=orbit_ppo task.orbit_target_mode=auto task.use_internal_mpc=false
        mv ./current_vs_energy.npz ./current_vs_energy.png $OUT_DIR
    fi

    # --- Step 3: CoT ---
    echo "[Step 3] Calculating CoT..."
    COT_OPTS="+eval.episodes=5 +eval.harsh_speed=0.2 +eval.current_dir='[1,0,0]' env.max_episode_length=400"

    if [ "$ALGO" == "mpc" ]; then
        ~/isaac410/python.sh scripts/evaluate_cot_nominal_harsh.py task=OrbitCylinder_MPC algo=ppo headless=true env.num_envs=1 mode=evaluate +eval.steps=4000 task.disturbances.evaluate.payload.enable_payload=true task.disturbances.evaluate.payload.mass='[0.1,0.1]' task.disturbances.evaluate.payload.z='[0.0,0.0]' +eval.episodes=5 +eval.harsh_speed=0.2 +eval.current_dir='[1,0,0]' env.max_episode_length=400 +eval.controller=internal_mpc task.use_pypose_mpc=false
        mv ./cot_nominal_harsh_mpc.npz ./cot_nominal_harsh_mpc.png $OUT_DIR
    elif [ "$ALGO" == "ppo" ]; then
        ~/isaac410/python.sh scripts/evaluate_cot_nominal_harsh.py task=OrbitCylinder_MPC_PPO algo=ppo headless=true env.num_envs=1 mode=evaluate +eval.steps=4000 task.disturbances.evaluate.payload.enable_payload=true task.disturbances.evaluate.payload.mass='[0.1,0.1]' task.disturbances.evaluate.payload.z='[0.0,0.0]' +eval.episodes=5 +eval.harsh_speed=0.2 +eval.current_dir='[1,0,0]' env.max_episode_length=400 +eval.ckpt="$CKPT"
        mv ./cot_nominal_harsh.npz ./cot_nominal_harsh.png $OUT_DIR
    elif [ "$ALGO" == "acmpc" ]; then
        ~/isaac410/python.sh scripts/evaluate_cot_nominal_harsh.py task=OrbitCylinder_MPC algo=ppo_pypose_cylinder_mpc_werr_wu_tv headless=true env.num_envs=1 mode=evaluate +eval.steps=4000 task.disturbances.evaluate.payload.enable_payload=true task.disturbances.evaluate.payload.mass='[0.1,0.1]' task.disturbances.evaluate.payload.z='[0.0,0.0]' +eval.episodes=5 +eval.harsh_speed=0.2 +eval.current_dir='[1,0,0]' env.max_episode_length=400 \
        +eval.controller=policy +eval.ckpt="$CKPT" task.reward_mode=orbit_ppo task.orbit_target_mode=auto task.use_internal_mpc=false
        mv ./cot_nominal_harsh.npz ./cot_nominal_harsh.png $OUT_DIR
    fi

    echo "Completed: $FOLDER_NAME. Saved to $OUT_DIR"
done
