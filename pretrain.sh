export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# python=/home/disk1/yiding/python_env/Python-2.7.8/bin/python
python=/home/disk2/yiding/research/GMVSAE/Python-3.6.5/bin/python3
gpu_id=7

${python} run_loop.py --mode=pretrain --cluster_num=5 --num_epochs=10 --gpu_id=${gpu_id} --learning_rate=0.001 --model_dir=./pretrain
