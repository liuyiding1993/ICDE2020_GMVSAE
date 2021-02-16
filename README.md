# Online Anomalous Trajectory Detection with Deep Generative Sequence Modeling (ICDE 2020)

## How To Use (the new version)

### Preprocessing
- Step1: Download Porto data (i.e., <tt>train.csv.zip</tt>) from https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data.
- Step2: Put the data file in <tt>./data/</tt>, and unzip it as <tt>porto.csv</tt>.
- Step3: Set the dataset info in <tt>./preprocess/preprocess.py</tt>), run preprocessing by <tt>sh preprocess.sh</tt>.

The processed data files (i.e., <tt>processed_porto_train.csv</tt> and <tt>processed_porto_val.csv</tt>) will be put in <tt>./data</tt>.


### Training 

#### Example of training on Porto dataset:
```python
python run_loop.py --mode=train --cluster_num=5 --num_epochs=5 --gpu_id=0 --model_dir=./ckpt --learning_rate=1e-4 --num_epochs=10 --pretrain_dir=./pretrain
```
T
More conveniently, we can run pretraining, training and evaluation via <tt>pretrain.sh</tt>, <tt>train.sh</tt> and <tt>eval.sh</tt>, respectively.

#### Parameters:
| Name                  | Type            | Description   |
| :-------------        |:-------------   |:------------- |
| mode                  | enum(str)       | pretrain, train or evaluate. |
| data_filename         | str             | data file (e.g., ./data/processed_porto.csv). |
| map_size              | \(int, int\)    | size of the grid map. |
| token_dim             | int             | dimensionality of grid token. | 
| rnn_dim               | int             | dimensionality of rnn hidden state. |
| cluster_num           | int             | number of Gaussian components. |
| model_dir             | str             | directory to save/load a model during training or eval. |
| pretrain_dir          | str             | directory to save/load a model during pretraining.  |
| num_negs              | int             | number of negative samples during training.  |
| optimizer             | enum(str)       | training optimizer (e.g., adam or sgd). |
| learning_rate         | float           | learning rate for training. |
| num_epochs            | int             | number of passes over the training data. |
| log_steps             | int             | number of batches to print the log info. |

## Citation

Please kindly cite the paper if this repo is helpful :)

```
@inproceedings{liu2020online,
  title={Online anomalous trajectory detection with deep generative sequence modeling},
  author={Liu, Yiding and Zhao, Kaiqi and Cong, Gao and Bao, Zhifeng},
  booktitle={2020 IEEE 36th International Conference on Data Engineering (ICDE)},
  pages={949--960},
  year={2020},
  organization={IEEE}
}
```
