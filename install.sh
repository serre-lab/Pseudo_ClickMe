sudo mkdir -p /mnt/disks/clickme-with-pseudo
sudo mount -o discard,defaults,noload  /dev/sdb /mnt/disks/clickme-with-pseudo

sudo mkdir -p /mnt/disks/bucket
sudo chmod -R 777 /mnt/disks/bucket
gcsfuse -o allow_other --file-mode=777 --dir-mode=777 serrelab /mnt/disks/bucket
# /mnt/disks/bucket/pseudo_clickme

sudo mkdir /tmp/tpu_logs
sudo chmod -R 777 /tmp/tpu_logs

sudo pip3 install --upgrade pip
sudo pip3 install timm==0.8.23.dev0 
sudo pip3 install wandb==0.15.0

sleep 5
python3 -m wandb login 486f67137c1b6905ac11b8caaaf6ecb276bfdf8e

sudo cp -f /clickme-with-pseudo/Pseudo_ClickMe/xla_dist.py /usr/local/lib/python3.8/dist-packages/torch_xla/distributed/
sudo cp -f /clickme-with-pseudo/Pseudo_ClickMe/cluster.py /usr/local/lib/python3.8/dist-packages/torch_xla/distributed/

export TPU_NAME=harmonization-pseudo
export XRT_TPU_CONFIG="localservice;0;localhost:51011"