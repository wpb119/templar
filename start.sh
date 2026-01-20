set -e

source .venv/bin/activate
export $(cat .env | sed 's/#.*//g' | xargs)

CPU_CORES=$(nproc)
THREADS_PER_PROCESS=$((CPU_CORES / 4))

export NCCL_SOCKET_IFNAME=lo
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_DEBUG=ERROR
export NCCL_LAUNCH_MODE=GROUP

export MASTER_ADDR=localhost
export MASTER_PORT=29500

export OMP_NUM_THREADS=$(( THREADS_PER_PROCESS < 64 ? THREADS_PER_PROCESS : 64 ))

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 neurons/miner.py \
 --netuid 3 \
 --wallet.name test-coldkey \
 --wallet.hotkey test-hotkey \
 --standalone \
 --subtensor.network finney \
 --sync_state \
 --logging.info