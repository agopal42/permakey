# PermaKey
This is the code repository complementing the ICLR 2021 paper. 

**Unsupervised Object Keypoint Learning using Local Spatial Predictability**  
Anand Gopalakrishnan, Sjoerd van Steenkiste, J&uuml;rgen Schmidhuber
https://arxiv.org/abs/2011.12930

## Dependencies:
Please install all the dependencies listed in the `requirements.txt` file. Additionally, please install the latest version of DeepMind's graph-nets toolbox using `pip install git+git://github.com/deepmind/graph_nets.git` 

## Dataset:
The dataset used for the unsupervised learning of object keypoints can be found [here](https://drive.google.com/file/d/1acothwSMI7ObOyZt3GfoMmrkjuTgHNT8/view?usp=sharing).

## Files:
* `preprocess.py`: contains data loader and utility functions for preprocessing the Atari image frames.
* `vision_modules.py`: contains all the neural network modules for keypoint learning i.e. either by our "permakey" or "transporter".
* `ul_loss.py`: contains functions for computing various loss terms used for keypoint learning i.e. either by our "permakey" or "transporter". 
* `pkey_train.py`: training script for "permakey" keypoints. Saves logs and model ckpts within folder "permakey_exp/".
* `transporter_train.py`: training script for "transporter" keypoints. Saves logs and model ckpts within folder "transporter_exp/".
* `viz.ipynb`: jupyter-notebook which contains functions/scipts used for all visualizations/plots shown in the paper.
* `agent.py`: contains all the neural networks modules used by the keypoint-based RL agent.
* `rl_loss.py`: contains functions used for computing the loss for the keypoints-based RL agent.
* `agent_train.py`: training script for keypoints-based RL agent.  


## Experiments:
Use the following commands to recreate the various experimental results in the paper.

### PermaKey on Atari games
To train the PermaKey model (as shown in Figure 2.a in the paper) on MsPacman envrionment, run:

`python pkey_train.py with env="mspacman" num_keypoints=7 data_dir="PATH/TO/DATASET_FOLDER"` 

Pass the path to dataset folder using the `data_dir` flag.

### PermaKey on Noisy Atari games
To train the PermaKey model on a Noisy MsPacman environment (as shown in Figure 2.b), use the flag `noise_type`:

`python pkey_train.py with env="mspacman" num_keypoints=7 noise_type="both" data_dir="PATH/TO/DATASET_FOLDER"`

The `noise_type` flag supports values "horizontal", "vertical", "both", "none" (default) for superimposing different synthetic distractor object types i.e. "moving" colored bars) on the Atari game frame.

All logs, hyperparameter configs and model ckpts of a run will be saved in a folder under the directory `permakey_exp`.

### Transporter on (Noisy) Atari games
To train the baseline "Transporter" model (as shown in Figure 2.a) on MsPacman envrionment, run: 

`python transporter_train.py with env="mspacman" num_keypoints=7 data_dir="PATH/TO/DATASET_FOLDER"`

Other games in the dataset can be trained by using the following value(s) for `env` flag ("frosbite", "seaquest", "battlezone", "space_invaders" or "enduro")
To train the baseline "Transporter" model (as shown in Figure 2.b) on the "Noisy" Atari envs, please use the `noise_type` flag as shown above.
All logs, hyperparameter configs and model ckpts will be saved in a folder under the directory `transporter_exp`.

### Ablations
Commands to run the ablation study experiments (as shown in section 5.3) are as follows:

#### Number of Keypoints
Use the flag `num_keypoints` to specify the number of keypoints in a PermaKey model. For example: 

`python pkey_train.py with env="frostbite" num_keypoints=20` 

#### Spatial Resolution of feature embedding
Use the flag `lsp_layers` to specify list of conv-encoder layer(s) chosen for LSP computation in a PermaKey model. For example:

`python pkey_train.py with env="space_invaders" num_keypoints=25 lsp_layers=0,1` 

### Visualizing Keypoints
After training both models (Permakey and Transporter) on MsPacman env as shown above. If you'd like to compare and visualize the keypoints learned, first run the command `compare_kpts` as shown below: 

`python pkey_train.py compare_kpts with env="mspacman" num_keypoints=7 tp_fname="MODEL_CKPT_FOLDER_NAME" tp_epoch=0 pkey_fname="MODEL_CKPT_FOLDER_NAME" pkey_epoch=0 ablation=False seed=123`.  

This command loads the 2 models from specified folder names & checkpoints, evaluates them on a shared test set (seed used for creating test split). It logs the test set results in a folder under the directory `compare_kpts` (use the flag `save_base_dir` to change it). For saving logs for ablation exps (shown in section 5.3) and visualizations, use the flag `ablation` as in the example above to indicate whether it's an evaluation for ablation(s) (True) or model comparison (False).    

Then use the appropriate visualization and plotting scripts available in the jupyter-notebook `viz.ipynb`. You will need to provide appropriate values for variables such as `logs_base_dir`: path to the base directory containing saved logs of models that you'd like to visualize, folder name under the directory `compare_kpts` containing saved logs of PermaKey and Transporter models and `batch`: batch id in test set you want to visualize.

### Keypoints-based RL agents
To train the keypoint-based RL agents, first we need to load the appropriate pre-trained keypoint model. Please specify appropriate values for flags `vis_ckpt_fname`: folder name (under the directories `permakey_exp` or `transporter_exp` containing requisite pre-trained keypoint model checkpoints and `vis_load`: checkpoint epoch of pre-trained model you wish to use for various (Noisy) Atari envs and number of keypoints configurations available.  

To train a "PKey-CNN" model (as shown in Table 1) run:

`python agent_train.py with mspacman kp_type="permakey" vis_ckpt_fname="VIS_CKPT_FOLDER_NAME"  vis_load=X`. 

This commands loads the pre-trained PermaKey keypoint model in the folder `VIS_CKPT_FOLDER_NAME` checkpointed at epoch `X` and trains a keypoint-based RL agent with a cnn keypoint-encoder network. 
Use `battlezone`, `seaquest` and/or `frostbite` instead of `mspacman` to train agents on the other envs shown in Table 1.

Model checkpoints, hyperparameter configs and logs are stored in a folder under the directory `rl_exp`. 

To train a "PKey-GNN" model (as shown in Table 1) use the flag `kpt_encoder_type` as shown below:

`python agent_train.py with mspacman kp_type="permakey" vis_ckpt_fname="VIS_CKPT_FOLDER_NAME" vis_load=X kpt_encoder_type="gnn"`.

To train a "Transporter (re-imp.)" model (as shown in Table 1) use the flag `kp_type` as shown below:

`python agent_train.py with mspacman kp_type="transporter" vis_ckpt_fname="VIS_CKPT_FOLDER_NAME" vis_load=X`

To train a "Transporter-GNN" model (as shown in Table 1) run:

`python agent_train.py with mspacman kp_type="transporter" kpt_encoder_type="gnn" vis_ckpt_fname="VIS_CKPT_FOLDER_NAME" vis_load=X`

To recreate results from Table 2. on "Noisy" Atari envs, please use the flag `noise_type` alongwith the commands shown above. Remember to use appropriate pre-trained keypoint models trained on "Noisy" Atari envs. For example,

`python agent_train.py with mspacman kp_type="permakey" noise_type="both" vis_ckpt_fname="VIS_CKPT_FOLDER_NAME" vis_load=X`

To recreate results from Table 3. on RL keypoints ablation, please use the flag `num_keypoints` alongwith the commands shown above. Again, remember to use appropriate pre-trained keypoint models with specific number of keypoints as in the RL ablation experiment.   

After training "Pkey-CNN" RL agents/policies on MsPacman env. To compute evaluation scores, run the `evaluate` command as shown below:

`python agent_train.py evaluate with mspacman kp_type="permakey" vis_ckpt_fname="VIS_CKPT_FOLDER_NAME" vis_load=X load_ckpts=A,B,C eval_seeds=1,2,3,4,5`  

Pass appropriate values for flags `load_ckpts`: list of checkpoint indices (integers) of the agents/policies used for evaluation and `eval_seeds`: list of environment seeds you'd like to use for evaluating the policies.

The keypoints-based RL agent training code `agent_train.py` supports distributed training using [horovod](https://horovod.readthedocs.io/en/stable/). If the default config value for batch size leads to memory issues on a single GPU card, the script uses data-parallelism to split the batch across multiple GPUs. For example use the command:

`horovodrun -np 2 localhost:2 python agent_train.py with mspacman kp_type="permakey" gpu=0,1`

Use the flag `gpu` to pass the list of GPU ids to run on. Please refer to the horovod documentation for more information on how to run distributed training using the 'horovodrun' command.



### Cite
If you make use of this code in your own work, please cite our paper:
```
@inproceedings{
gopalakrishnan2021unsupervised,
title={Unsupervised Object Keypoint Learning using Local Spatial Predictability},
author={Anand Gopalakrishnan and Sjoerd van Steenkiste and J{\"u}rgen Schmidhuber},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=GJwMHetHc73}
}
```
