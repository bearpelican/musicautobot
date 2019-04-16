#!/usr/bin/env python

# Running on AWS
# python launch.py --aws
#
# Wikitext-3 training. 3 hours per epoch.
#
# /ncluster/runs/fastai.wikitext-3-raw
# Better model found at epoch 1 with val_loss value: 3.7706105709075928.
# 2         3.793330    3.583894    0.378477  2:53:53
# Better model found at epoch 2 with val_loss value: 3.5838944911956787.
# 3         3.667973    3.432626    0.395862  2:54:01
# Better model found at epoch 3 with val_loss value: 3.4326255321502686.
# 4         3.608931    3.378200    0.402312  2:54:03
# Better model found at epoch 4 with val_loss value: 3.3782002925872803.
# Total time: 14:32:22

# Running locally with wikitext-2
# mkdir -p  ~/data/wikitext-2-raw
# aws s3 cp s3://yaroslavvb2/data/wikitext-2-raw/data_save.pkl ~/data/wikitext-2-raw
# (https://s3-us-east-1.amazonaws.com/yaroslavvb2/data/wikitext-2-raw/data_save.pkl)
# python launch.py

import argparse
import ncluster
import os

IMAGE_NAME = 'midi_generator_v10'
INSTANCE_TYPE = 'p3.2xlarge'
NUM_GPUS = {'p3.2xlarge': 1, 'p3.8xlarge':4, 'p3.16xlarge':8}[INSTANCE_TYPE]

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='txl',
                    help="name of the current run, used for machine naming and tensorboard visualization")
parser.add_argument('--machines', type=int, default=1,
                    help="how many machines to use")
parser.add_argument("--local", action="store_true", help="enable to run on AWS")
args = parser.parse_args()

if not args.local: ncluster.set_backend('aws')


# routines to build NCCL ring orders
def get_nccl_params(num_tasks, num_gpus):
  if num_tasks <= 1:
    return 'NCCL_DEBUG=VERSION'
  return 'NCCL_MIN_NRINGS=4 NCCL_SINGLE_RING_THRESHOLD=10 NCCL_DEBUG=VERSION'


def format_params(arg):
  if isinstance(arg, list) or isinstance(arg, dict):
    return '\"' + str(arg) + '\"'
  else:
    return str(arg)


def main():
  supported_regions = ['us-west-2', 'us-east-1', 'us-east-2', 'local']
  assert ncluster.get_region() in supported_regions, f"required AMI {IMAGE_NAME} has only been made available in regions {supported_regions}, but your current region is {ncluster.get_region()}"

  job = ncluster.make_job(name=args.name,
                          run_name=f"{args.name}-{args.machines}",
                          num_tasks=args.machines,
                          image_name=IMAGE_NAME,
                          instance_type=INSTANCE_TYPE,
                          )

  job.upload('src')
  job.upload('scripts')
  job.run(f'conda activate midi')
  job.run('cd scripts')

  nccl_params = get_nccl_params(args.machines, NUM_GPUS)

  # Training script args
  default_params = [
      # '--load', f'/ncluster/models/{args.name}.pth',
    '--path', '~/data/midi/v10/midi_encode/'
      ]
  params = [
    '--save', f'large_single/lq/1_ep44',
    '--cache', 'tmp/lq',
    '--batch_size', '8',
    '--large',
    '--single_stream',
    '--epochs', '44',
    '--lr', '.008'
  ]
  training_params = default_params + params
  training_params = ' '.join(map(format_params, training_params))
  train_script = 'run_txl_npenc.py'


  # TODO: simplify args processing, or give link to actual commands run
  for i, task in enumerate(job.tasks):
    
    dist_params = f'--nproc_per_node={NUM_GPUS} --nnodes={args.machines} --node_rank={i} --master_addr={job.tasks[0].ip} --master_port={6006}'
    cmd = f'{nccl_params} python -m torch.distributed.launch {dist_params} {train_script} {training_params}'
    # task.run(f'echo {cmd} > {job.logdir}/task-{i}.cmd')  # save command-line
    task.run(cmd, non_blocking=True)

#   print(f"Logging to {job.logdir}")


if __name__ == '__main__':
  main()