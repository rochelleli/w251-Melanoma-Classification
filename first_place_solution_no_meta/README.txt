#RUNNING TRAIN on EC2 Instance (I thought I used nvidia container but I mispoke, that was just for when getting prediction working on Jetson)
# I am verifying now... Sorry if that lead to some confusion.
# I am retracing my steps now and will publish here

Launch Instance

Search:  Nvidia Deep Learning AMI

Select:  NVIDIA Deep Learning AMI

Choose Instance Type:

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


Add Storage:  400GB?

Leave rest Deafults and Launch - Choose Existing Key Pair you have.  use one that I have configured with my Local Putty.

Retrieve the IP 

Save IP into your Putty Startup

Download Final Project NVIDA Container

upload melanoma-256.zip
unzip to /data directory

#MY EXMAMPLE
# /home/ubuntu/fpdata
#	|
#	-- melanoma-256
#	    |
#		-- train
#		-- test
#		-- ...
# /home/ubuntu/w251/fp
#	|
#	-- python files

time sudo docker run -it --rm --net=host --gpus all  --privileged -v /home/ubuntu/fpdata:/data -v /home/ubuntu/w251/fp:/media joslateriii/melanoma

#You should get a docker root prompt

cd to /media

# Execute train python command
# To Train the Data run this
python3 train.py --kernel-type 9c_meta_b3_768_512_ext_18ep --data-dir /data/melanoma-256 --data-folder 768 --image-size 256 --enet-type efficientnet_b3 --use-meta --n-epochs 18 --use-amp --CUDA_VISIBLE_DEVICES 0,1

# RUNNING ON JETSON

sudo docker run -it --rm --runtime nvidia --network host -v /home/joslateriii/fp_data:/data -v /home/joslateriii/w251:/media nvcr.io/nvidia/l4t-ml:r32.6.1-py3

# To Run a Prediction run this
python3 predict.py --kernel-type 9c_meta_b3_768_512_ext_18ep --data-dir /data/melanoma-256 --data-folder 768 --image-size 512 --enet-type efficientnet_b3 --use-meta



