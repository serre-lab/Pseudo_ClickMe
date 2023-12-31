sudo mkdir -p /mnt/disks/clickme-with-pseudo
sudo mount -o discard,defaults,noload  /dev/sdb /mnt/disks/clickme-with-pseudo

sudo mkdir -p /mnt/disks/bucket
sudo chmod -R 777 /mnt/disks/bucket
gcsfuse -o allow_other --file-mode=777 --dir-mode=777 serrelab /mnt/disks/bucket
# /mnt/disks/bucket/pseudo_clickme

sudo mkdir /tmp/tpu_logs
sudo chmod -R 777 /tmp/tpu_logs

sudo pip3 install --upgrade pip
sudo pip3 install timm==0.9.0
sudo pip3 install wandb==0.15.0
sudo pip3 install torchmetrics==1.0.1
sudo pip3 install pathlib==0.2.0
sudo pip3 install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'

sleep 5
python3 -m wandb login 486f67137c1b6905ac11b8caaaf6ecb276bfdf8e

sudo cp -f /clickme-with-pseudo/Pseudo_ClickMe/xla_dist.py /usr/local/lib/python3.8/dist-packages/torch_xla/distributed/
sudo cp -f /clickme-with-pseudo/Pseudo_ClickMe/cluster.py /usr/local/lib/python3.8/dist-packages/torch_xla/distributed/

export TPU_NAME=harmonization-pseudo
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export XLA_USE_BF16=1 && export PYTHONUNBUFFERED=1

# python3 /clickme-with-pseudo/Pseudo_ClickMe/main_obj_recognition.py 
# python3 /clickme-with-pseudo/Pseudo_ClickMe/main_obj_recognition.py \
# -dd "/mnt/disks/clickme-with-pseudo/" \
# -mn "resnet50" \
# -md "imagenet" \
# -ep 90 \
# -bs 256 \
# -iv 100 \
# -lu 50 \
# -ev False \
# -pt False \
# -rs False \
# -gt True \
# -wb True
# python3 /clickme-with-pseudo/Pseudo_ClickMe/main_obj_recognition.py --dir '/mnt/disks/clickme-with-pseudo/test' --epoch 90 --pretrained True --eval False --model "imagenet"
# sudo python3 -m torch_xla.distributed.xla_dist --tpu=${TPU_NAME} --restart-tpuvm-pod-server --env XRT_MESH_CONNECT_WAIT=12000 --env XLA_USE_BF16=1 --env PYTHONUNBUFFERED=1 -- python3 /clickme-with-pseudo/Pseudo_ClickMe/main_obj_recognition.py