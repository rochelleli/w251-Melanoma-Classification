# To Train the Data run this
python3 train.py --kernel-type 9c_meta_b3_768_512_ext_18ep --data-dir /data/melanoma-256 --data-folder 768 --image-size 256 --enet-type efficientnet_b3 --use-meta --n-epochs 18 --use-amp --CUDA_VISIBLE_DEVICES 0,1

# To Run a Prediction run this
python3 predict.py --kernel-type 9c_meta_b3_768_512_ext_18ep --data-dir /data/melanoma-256 --data-folder 768 --image-size 512 --enet-type efficientnet_b3 --use-meta


