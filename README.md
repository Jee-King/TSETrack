# TSETrack
The official implementation for the paper [_Efficient Vision Transformer with Token Sparsification for Event-based Object Tracking_](https://xxx).

[[Models]([https://xx](https://1drv.ms/f/s!AoopRFuuZ7xoh9BxI6aZVJJwpHX5iw?e=Fx1qpv))][[Raw Results](https://1drv.ms/u/s!AoopRFuuZ7xoh9BvNK1eTCL_lX8jIg?e=t3froa)]



## Install the environment
**Option1**: Use the Anaconda (CUDA 10.2)
```
conda create -n tsetrack python=3.8
conda activate tsetrack
bash install.sh
```

**Option2**: Use the Anaconda (CUDA 11.3)
```
conda env create -f tsetrack_cuda113_env.yaml
```
 
## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Download FE240, VisEvent, COESOT dataset.
Put the tracking datasets in ./data. It should look like this:
   ```
   ${PROJECT_ROOT}
    -- data
        -- FE240
            |-- airplane_0
            |-- airplane_0
            ...
        -- VisEvent
            |-- train_subset
            |-- test_subset
        -- COESOT_dataset
            |-- training_subset
            |-- testing_subset
   ```


## Training
Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it under `$PROJECT_ROOT$/pretrained_models` (different pretrained models can also be used, see [MAE](https://github.com/facebookresearch/mae) for more details).

```
python tracking/train.py --script tsetrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
```

Replace `--config` with the desired model config under `experiments/tsetrack`. We use [wandb](https://github.com/wandb/client) to record detailed training logs, in case you don't want to use wandb, set `--use_wandb 0`.


## Evaluation
Download the model weights from [OneDrive]([https://xx](https://1drv.ms/f/s!AoopRFuuZ7xoh9BxI6aZVJJwpHX5iw?e=Fx1qpv)) 

Put the downloaded weights on `$PROJECT_ROOT$/output/checkpoints/train/tsetrack`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Testing examples:
```
python tracking/test.py tsetrack vitb_256_mae_ce_32x4_ep300 --dataset eotb --sequence val --threads 1 --num_gpus 1 --debug 0
```


## Visualization or Debug 
[Visdom](https://github.com/fossasia/visdom) is used for visualization. 
1. Alive visdom in the server by running `visdom`:

2. Simply set `--debug 1` during inference for visualization, e.g.:
```
python tracking/test.py tsetrack vitb_384_mae_ce_32x4_ep300 --dataset eotb --threads 1 --num_gpus 1 --debug 1
```
3. Open `http://localhost:8097` in your browser (remember to change the IP address and port according to the actual situation).

4. Then you can visualize the candidate elimination process.

## Acknowledgments
* Thanks for the [OSTrack](https://github.com/botaoye/OSTrack) library, which helps us to quickly implement our ideas. 

<!-- 
## Citation
If our work is useful for your research, please consider citing:

```Bibtex

``` -->

