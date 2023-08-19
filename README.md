# TPComplEx 
(Accepted by Expert Systems with Applications)

## Installation
Create a conda environment with pytorch and scikit-learn :
```
conda create --name tkbc_env python=3.7
source activate tkbc_env
conda install --file requirements.txt -c pytorch
```

Then install the kbc package to this environment
```
python setup.py install
```

## Datasets

To download the datasets, go to the ./tkbc/scripts folder and run:
```
chmod +x download_data.sh
./download_data.sh
```

GDELT dataset can be download [here](https://github.com/BorealisAI/de-simple/tree/master/datasets/gdelt) and rename the files without ".txt" suffix.

Once the datasets are downloaded, add them to the package data folder by running :
```
python tkbc/process_icews.py
python tkbc/process_yago.py
python tkbc/process_gdelt.py
```

This will create the files required to compute the filtered metrics.

## Reproducing results

Run the following commands to reproduce the results

```
CUDA_VISIBLE_DEVICES=0 python tkbc/learner.py --dataset ICEWS14 --model TPComplEx --rank 1594 --emb_reg 1e-1 --time_reg 1e-4 

CUDA_VISIBLE_DEVICES=0 python tkbc/learner.py --dataset ICEWS05-15 --model TPComplEx --rank 886 --emb_reg 1e-2 --time_reg 1e-2  

CUDA_VISIBLE_DEVICES=0 python tkbc/learner.py --dataset yago15k --model TPComplEx --rank 1892 --no_time_emb --emb_reg 1e-1 --time_reg 1e-4

CUDA_VISIBLE_DEVICES=0 python tkbc/learner.py --dataset gdelt --model TPComplEx --rank 1256 --emb_reg 1e-5 --time_reg 1e-2 

```

## Acknowledgement
We refer to the code of [TComplEx](https://github.com/facebookresearch/tkbc). Thanks for their contributions.

