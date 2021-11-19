#RUNNING TRAIN on EC2 Instance (I thought I used nvidia container but I mispoke, that was just for when getting prediction working on Jetson)
# I am verifying now... Sorry if that lead to some confusion.
# I am retracing my steps now and will publish here

Launch Instance

Search:  Nvidia Deep Learning AMI

Select:  NVIDIA Deep Learning AMI

Choose Instance Type:  I RECOMMEND g4dn8x at least - then you can do batchsize 64 and num workers = 4
- If you have the $$ then go even bigger and you can bump up batchsize to 128 and 32 workers.

I tried 2x and it didn't have enough memory 

Instance Type	Software	EC2	Total
g4dn.xlarge	$0.00	$0.526	$0.526/hr
g4dn.2xlarge	$0.00	$0.752	$0.752/hr
g4dn.4xlarge	$0.00	$1.204	$1.204/hr
g4dn.8xlarge	$0.00	$2.176	$2.176/hr
g4dn.12xlarge	$0.00	$3.912	$3.912/hr
g4dn.16xlarge	$0.00	$4.352	$4.352/hr
p3.2xlarge	$0.00	$3.06	$3.06/hr
p3.8xlarge	$0.00	$12.24	$12.24/hr
p3.16xlarge	$0.00	$24.48	$24.48/hr
p3dn.24xlarge	$0.00	$31.212	$31.212/hr
p4d.24xlarge	$0.00	$32.773	$32.773/hr


Add Storage:  100GB should be fine

Leave rest Deafults and Launch - Choose Existing Key Pair you have.  use one that I have configured with my Local Putty.

Retrieve the IP 

Save IP into your Putty Startup Config

SSH to EC2 Instance

Clone our git Hub repository


ubuntu@ip-172-31-0-245:~$ git clone https://github.com/rochelleli/w251-Melanoma-                                                                         Classification.git
Cloning into 'w251-Melanoma-Classification'...
Username for 'https://github.com': johnslater3
Password for 'https://johnslater3@github.com':
remote: Enumerating objects: 44, done.
remote: Counting objects: 100% (44/44), done.
remote: Compressing objects: 100% (36/36), done.
remote: Total 44 (delta 19), reused 15 (delta 4), pack-reused 0
Unpacking objects: 100% (44/44), 273.10 KiB | 5.15 MiB/s, done.
ubuntu@ip-172-31-0-245:~$ ls
README.txt  mnist_example.sh  w251-Melanoma-Classification
ubuntu@ip-172-31-0-245:~$ cd w251-Melanoma-Classificatio
-bash: cd: w251-Melanoma-Classificatio: No such file or directory
ubuntu@ip-172-31-0-245:~$ cd w251-Melanoma-Classification/
ubuntu@ip-172-31-0-245:~/w251-Melanoma-Classification$ ls -lr
total 8
drwx------ 2 ubuntu ubuntu 4096 Nov 19 18:13 first_place_solution_no_meta
-rw------- 1 ubuntu ubuntu   31 Nov 19 18:13 README.md
ubuntu@ip-172-31-0-245:~/w251-Melanoma-Classification$ cd first_place_solution_n                                                                         o_meta/
ubuntu@ip-172-31-0-245:~/w251-Melanoma-Classification/first_place_solution_no_me                                                                         ta$ ls -lrt
total 336
-rw------- 1 ubuntu ubuntu   1077 Nov 19 18:13 util.py
-rw------- 1 ubuntu ubuntu   8970 Nov 19 18:13 train.py
-rw------- 1 ubuntu ubuntu    328 Nov 19 18:13 requirements.txt
-rw------- 1 ubuntu ubuntu   4775 Nov 19 18:13 predict.py
-rw------- 1 ubuntu ubuntu   3289 Nov 19 18:13 models.py
-rw------- 1 ubuntu ubuntu 278587 Nov 19 18:13 figure1.png
-rw------- 1 ubuntu ubuntu   7945 Nov 19 18:13 evaluate.py
-rw------- 1 ubuntu ubuntu    730 Nov 19 18:13 ensemble.py
-rw------- 1 ubuntu ubuntu   4197 Nov 19 18:13 dataset.py
-rw------- 1 ubuntu ubuntu    399 Nov 19 18:13 cam.py
-rw------- 1 ubuntu ubuntu   2081 Nov 19 18:13 README.txt
ubuntu@ip-172-31-0-245:~/w251-Melanoma-Classification/first_place_solution_no_meta$                                                                      


PSFTP to EC2 Instance and upload melanoma-256.zip

C:\Users\josla>psftp -i "C:\_W251\HW5\ec2_priv_key.ppk" ubuntu@<IP of EC2 Instance>
The server's host key is not cached in the registry. You
have no guarantee that the server is the computer you
think it is.
The server's ssh-ed25519 key fingerprint is:
ssh-ed25519 256 c7:99:f7:d8:40:3c:d1:13:6a:b7:61:97:7e:77:e7:13
If you trust this host, enter "y" to add the key to
PuTTY's cache and carry on connecting.
If you want to carry on connecting just once, without
adding the key to the cache, enter "n".
If you do not trust this host, press Return to abandon the
connection.
Store key in cache? (y/n) y
Using username "ubuntu".
Passphrase for key "rsa-key-20210923":
Remote working directory is /home/ubuntu
psftp>

unzip to /data directory

your directory structure should look like this

#MY EXMAMPLE
# /home/ubuntu/data
#	|
#	-- train
#	-- test
#	-- test.csv
#	-- train.csv
#   -- sampl_submission.csv
#	-- melanoma-256.zip
#
# /home/ubuntu/w251-Melanoma-Classification
#	|
#	-- first_place_solution_no_meta
#		|
#		-- util.py
#		-- train.py
#	    -- predict.py
#		-- models.py
#		-- figure1.png
#		-- evalute.py
#		-- ensemble.py
#		-- dataset.py
#		-- cam.py
#		-- README.txt
#	-- README.md

#Run this Docker command to download and run melanoma Container
# This maps volumes from container to your python files directory and your data
# This takes a couple minutes to download first time 
# When done you should have root@ip-Address:/fp prompt

time sudo docker run -it --rm --net=host --gpus all  --privileged -v /home/ubuntu/data:/data -v /home/ubuntu/w251-Melanoma-Classification/first_place_solution_no_meta:/melanoma joslateriii/melanoma

#You should get a docker root prompt

cd to /melanoma 

# Execute train python command
# To Train the Data run this
# I think --data-folder 768 is not used
# the kernel-type parameter is used as the name for which the weights are saved and it gets parsed for some runtime parameters

PARAMETERS:
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='/raid/')
    parser.add_argument('--data-folder', type=int, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--init-lr', type=float, default=3e-5)
    parser.add_argument('--out-dim', type=int, default=9)
    parser.add_argument('--n-epochs', type=int, default=15)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--fold', type=str, default='0,1,2,3,4')


python3 train.py --kernel-type johns-model-test-1 --data-dir /data --data-folder 256 --batch-size 128 --num-workers 4 --image-size 256 --enet-type efficientnet_b3 --use-meta --n-epochs 18 --use-amp --CUDA_VISIBLE_DEVICES 0,1


Running successfully should look like this:
root@ip-172-31-0-245:/melanoma# python3 train.py --kernel-type johns-model-test-1 --data-dir /data --data-folder 256 --batch-size 64 --num-workers 4 --image-size 256 --enet-type efficientnet_b3 --use-meta --n-epochs 18 --use-amp --CUDA_VISIBLE_DEVICES 0,1
Number of CUDA DEVICES
True
Parallel call
26161 6531
Fri Nov 19 21:03:49 2021 Fold 0, Epoch 1
loss: 1.42720, smth: 1.87160:  13%|#######9                                                   | 55/409 [00:45<04:39,  1.27it/s]


# RUNNING ON JETSON
# I haven't validated these instructions yet 

sudo docker run -it --rm --runtime nvidia --network host -v /home/joslateriii/fp_data:/data -v /home/joslateriii/w251:/media nvcr.io/nvidia/l4t-ml:r32.6.1-py3

# To Run a Prediction run this
python3 predict.py --kernel-type 9c_meta_b3_768_512_ext_18ep --data-dir /data/melanoma-256 --data-folder 768 --image-size 512 --enet-type efficientnet_b3 --use-meta


