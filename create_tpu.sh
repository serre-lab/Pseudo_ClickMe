echo "\n\n"
echo "\t Trying to start tpu vms: (harmonization-pseudo), zone: (us-central1-a) tpu: (v3-8)"
echo "\n\n"

# create tpu by specifying the name of the vm, zone, tpu device
gcloud alpha compute tpus tpu-vm create harmonization-pseudo --zone=us-central1-a --accelerator-type=v3-8 --version=tpu-vm-pt-1.10 --preemptible
sleep 2

# attach disk
gcloud alpha compute tpus tpu-vm attach-disk harmonization-pseudo --disk=clickme-with-pseudo --zone=us-central1-a --mode=read-only
sleep 3

# ssh
gcloud alpha compute tpus tpu-vm ssh harmonization-pseudo --zone=us-central1-a --worker=all --command="sudo mkdir /clickme-with-pseudo/
sudo chmod 777 /clickme-with-pseudo/
cd /clickme-with-pseudo/
git clone https://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@github.com/serre-lab/Pseudo_ClickMe.git
cd Pseudo_ClickMe/
sh install.sh"