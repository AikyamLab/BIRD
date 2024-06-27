# Towards Bridging the Gap for Fairness in Knowledge Distillation

## Installation
Install the necessary packages with the following command:<br>
`pip install -r requirements.txt`

## Data preparation

#### CelebA
Download the CelebA dataset (img align and crop version) from the official <a href="https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html">website</a> to a desired directory. Follow the steps described in `PreprocessData.ipynb` to prepare a <b>fair</b> test benchamark from the original test split following previous works.


### Training and evaluating
#### CelebA
For a quick start please use the scripts provided in `scripts/` directory. Run all scripts from the root directory of this repository. 

1. Train baseline student and teacher models:<br>
   ` bash ./scripts/celeba/clip.sh ` <br>
   ` bash ./scripts/celeba/clip50.sh ` <br>
   ` bash ./scripts/celeba/flava.sh ` <br>
   ` bash ./scripts/celeba/res18.sh ` <br>
   ` bash ./scripts/celeba/res34.sh ` <br>
   ` bash ./scripts/celeba/shuffv2.sh ` <br>

2. Run KD baselines: <br>
	a. BKD <br>
   		`bash scrips/celeba/kd/clip.sh`<br>
   		`bash scrips/celeba/kd/clip50.sh`<br>
   		`bash scrips/celeba/kd/flava.sh`<br>
   		`bash scrips/celeba/kd/res18.sh`<br>
   		`bash scrips/celeba/kd/res34.sh`<br>
   		`bash scrips/celeba/kd/shuffv2.sh`<br>
	b. FitNet Stage 1 <br>
		`bash scrips/celeba/fit-s1/clip.sh`<br>
   		`bash scrips/celeba/fit-s1/clip50.sh`<br>
   		`bash scrips/celeba/fit-s1/flava.sh`<br>
   		`bash scrips/celeba/fit-s1/res18.sh`<br>
   		`bash scrips/celeba/fit-s1/res34.sh`<br>
   		`bash scrips/celeba/fit-s1/shuffv2.sh`<br>
	b. FitNet Stage 2 <br>
		`bash scrips/celeba/fit-s2/clip.sh`<br>
   		`bash scrips/celeba/fit-s2/clip50.sh`<br>
   		`bash scrips/celeba/fit-s2/flava.sh`<br>
   		`bash scrips/celeba/fit-s2/res18.sh`<br>
   		`bash scrips/celeba/fit-s2/res34.sh`<br>
   		`bash scrips/celeba/fit-s2/shuffv2.sh`<br>
	c. AT <br>
   		`bash scrips/celeba/AT/res18.sh`<br>
   		`bash scrips/celeba/AT/res34.sh`<br>
   		`bash scrips/celeba/AT/shuffv2.sh`<br>
	d. AD <br>
		`bash scrips/celeba/AD/clip.sh`<br>
   		`bash scrips/celeba/AD/clip50.sh`<br>
   		`bash scrips/celeba/AD/flava.sh`<br>
   		`bash scrips/celeba/AD/res18.sh`<br>
   		`bash scrips/celeba/AD/res34.sh`<br>
   		`bash scrips/celeba/AD/shuffv2.sh`<br>
	e. MFD <br>
		`bash scrips/celeba/mmdv2/clip.sh`<br>
   		`bash scrips/celeba/mmdv2/clip50.sh`<br>
   		`bash scrips/celeba/mmdv2/flava.sh`<br>
   		`bash scrips/celeba/mmdv2/res18.sh`<br>
   		`bash scrips/celeba/mmdv2/res34.sh`<br>
   		`bash scrips/celeba/mmdv2/shuffv2.sh`<br>

3. Run BIRD (Our Method):<br>
    `bash scrips/celeba/bird/v1/clip.sh`<br>
    `bash scrips/celeba/bird/v1/clip50.sh`<br>
    `bash scrips/celeba/bird/v1/flava.sh`<br>
    `bash scrips/celeba/bird/v1/res18.sh`<br>
    `bash scrips/celeba/bird/v1/res34.sh`<br>
    `bash scrips/celeba/bird/v1/shuffv2.sh`<br>

<u>Notes to the users</u>:<br>
1. The above scripts will run 5 independent runs of each method and log results on the wandb server. To collect average results, kindly download and parse the wandb project page using Pandas.
2. In case running the scripts parallely incurs a cuda out of memory error, remove the `&` flag at the end of each bash command to run the models sequentially.
3. BIRD takes an additional training step time due to <a href="https://github.com/facebookresearch/higher">higher</a> dependencies. We plan on fixing the train time optimization subsequently.
4. For the remaining experiments change the model flags within the provided scripts to run.





