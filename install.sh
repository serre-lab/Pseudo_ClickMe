sudo mkdir -p /mnt/disks/clickme-with-pseudo
sudo mount -o discard,defaults,noload  /dev/sdb /mnt/disks/clickme-with-pseudo

sudo pip3 install --upgrade pip
sudo pip3 install timm==0.8.23.dev0 

sleep 5
python3 -m wandb login 486f67137c1b6905ac11b8caaaf6ecb276bfdf8e

export XRT_TPU_CONFIG="localservice;0;localhost:51011"