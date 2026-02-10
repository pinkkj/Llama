set -euo pipefail

cd /PROJECT/0990910002_A/Megatron-LLaMA
export PYTHONPATH=/PROJECT/0990910002_A/Megatron-LLaMA

# === COMM / DEBUG ===
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# === NETWORK IFACE 고정 (중요: loopback 방지) ===
# 노드 간 TCP 제어(Gloo) + NCCL 소켓이 사용할 NIC
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export TP_SOCKET_IFNAME=bond0

# === IB ON ===
export NCCL_IB_DISABLE=0
# 아래 2개는 환경에 맞게 조정
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=2

# === DATA ===
DATA_PREFIX="/PROJECT/0990910002_A/dataset/NIA_final/univa_nia_corpus__v0.1.0-mt/univa_nia_corpus__v0.1.0_text_document"

# === TOKENIZER ===
TOKENIZER_PATH="/PROJECT/0990910002_A/hf_models/Llama-2-13b-hf"

# === OUTPUT ===
SAVE_DIR="/PROJECT/0990910002_A/checkpoints/llama13b_nia_durability"
mkdir -p "${SAVE_DIR}"

# === TRAINING LENGTH ===
TRAIN_ITERS=64000

# === MODEL: LLaMA 13B ===
VOCAB_SIZE=32000

# === PARALLEL ===
TP=4
PP=1

# === BATCHING ===
SEQ_LEN=2048
MICRO_BS=4
GLOBAL_BS=256

# === REQUIRED ENV ===
: "${MASTER_ADDR:?set MASTER_ADDR}"
: "${MASTER_PORT:?set MASTER_PORT}"
: "${NODE_RANK:?set NODE_RANK}"

echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NODE_RANK=${NODE_RANK}"
echo "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"

torchrun \
  --nnodes=4 \
  --nproc_per_node=8 \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  pretrain_llama.py \
  --tensor-model-parallel-size ${TP} \
  --pipeline-model-parallel-size ${PP} \
  --num-layers 40 \
  --hidden-size 5120 \
  --num-attention-heads 40 \
  --seq-length ${SEQ_LEN} \
  --max-position-embeddings ${SEQ_LEN} \
  --micro-batch-size ${MICRO_BS} \
  --global-batch-size ${GLOBAL_BS} \
  --train-iters ${TRAIN_ITERS} \
  --lr 3e-4 \
  --min-lr 3e-5 \
  --lr-decay-style cosine \
  --lr-warmup-iters 2000 \
  --weight-decay 0.1 \
  --clip-grad 1.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --init-method-std 0.02 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --swiglu \
  --use-rotary-position-embeddings \
  --untie-embeddings-and-output-weights \
  --tokenizer-type PretrainedFromHF \
  --tokenizer-name-or-path ${TOKENIZER_PATH} \
  --vocab-size ${VOCAB_SIZE} \
  --data-path ${DATA_PREFIX} \
  --data-impl mmap \
  --split 100,0,0 \
  --log-interval 10 \
  --save-interval 20000 \
  --eval-interval 1000000000 \
  --eval-iters 0 \
  --save ${SAVE_DIR} \
  --bf16 \
  --use-distributed-optimizer
