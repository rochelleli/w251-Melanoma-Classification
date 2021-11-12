sudo docker run -it --rm --runtime nvidia --network host -v /home/joslateriii/fp_data:/data -v /home/joslateriii/w251:/media nvcr.io/nvidia/l4t-ml:r32.6.1-py3

sudo apt-get install nvidia-opencv python3-numpy python3-scipy
LANG=en_US.UTF-8 pip3 install --no-deps scikit-image # THIS ONE TAKES A WHILE
pip3 install --no-deps albumentations
pip3 install --no-deps qudida
pip3 install geffnet
pip3 install resnest
pip3 install pretrainedmodels

# To Train the Data run this
python3 train.py --kernel-type 9c_meta_b3_768_512_ext_18ep --data-dir /data/melanoma-256 --data-folder 768 --image-size 256 --enet-type efficientnet_b3 --use-meta --n-epochs 18 --use-amp --CUDA_VISIBLE_DEVICES 0,1

# To Run a Prediction run this
python3 predict.py --kernel-type 9c_meta_b3_768_512_ext_18ep --data-dir /data/melanoma-256 --data-folder 768 --image-size 512 --enet-type efficientnet_b3 --use-meta


