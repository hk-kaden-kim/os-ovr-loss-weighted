# OpenSet OvR Model with Loss Weighting

This repository has an implementation of experiments performed in the master thesis "Improved Losses for One-vs-Rest Classifiers in Open-Set Problems" by Hyeongkyun Kim at AIML Group at the University of Zürich in 2024. You can find a paper in [here]().

``` 
@mastersthesis{kim2024opensetovr,
  title={Improved Losses for One-vs-Rest Classifiers in Open-Set Problems},
  author={Hyeongkyun Kim},
  school= {AIML Group, University of Zurich},
  year={2024}
}
 ```

## License
This code package is open-source based on the BSD license. Please see LICENSE for details.

## Environment Setup
You can install all the dependencies by running a Conda installation script, given with:

```
conda env create -f environment.yaml
```

After that, please run the environment:

```
conda activate os-ovr-loss-weighted
```

## Dataset Setup

### for Small-Scale
If you do not have EMNIST dataset, please run the code below:
```
python ./data/SmallScale/emnist_setup.py -O [outdir]
```
Afterward, set ***data.smallscale.root*** in ```train.yaml``` to the same with ```[outdir]```.

### for Large-Scale
If you do not have the ImageNet dataset, please follow the guideline [here](https://github.com/AIML-IfI/openset-imagenet/tree/main?tab=readme-ov-file#data).

Afterward, set ***data.largescale.root*** in ```train.yaml``` to the folder path that ends with ```.../ImageNet/ILSVRC2012```.

The pre-computed protocol files are at ```./data/LargeScale/protocols```. 

## Training

The model can be trained using ```training.py``` with the configuration file ``` ./config/train.yaml ```.

```
python -u ./training.py -s [seed] -g [gpu] &> training.output &
```
where  ```seed``` the random seed and ```gpu``` can be used when GPU is available. Trained models are saved in ``` ./models ```. You can use it to reproduce results.

## Evaluation

Model Evaluation can be performed using ```evaluation.py``` and the configuration file ``` ./config/eval.yaml ```.

```
python -u ./evaluation.py -s [seed] -g [gpu] &> evaluation.output &
```
where ```seed``` the random seed and ```gpu``` can be used when GPU is available. After running, move the evaluation results to ``` ./results ``` for the following analysis. 

## Analysis
All analyses are based on the results saved in ``` ./results ```. You can choose which results are loaded via ``` ./library/evals/constants.py  ```

### for Small-Scale
Evaluation results and related plots can be found in ``` results-small.ipynb ```.

### for Large-Scale
Evaluation results and related plots can be found in ``` results-large.ipynb ```.