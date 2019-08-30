QUERY="$(nvidia-smi --query-gpu=gpu_name --format=csv | wc -l)"
NUM_GPUS=$((QUERY-1))

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} ${SCRIPT} "$@"
#python -m torch.distributed.launch --nproc_per_node=1 ${SCRIPT} "$@"
