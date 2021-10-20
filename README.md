# IRM-when it works and when it doesn't: A test case of natural language inference
Accepted as a conference paper for NeurIPS 2021

>**Abstract**: Invariant Risk Minimization (IRM) is a recently proposed framework for out-of-distribution (o.o.d) generalization.  Most of the studies on IRM so far have focused on theoretical results, toy problems, and simple models. In this work, we investigate the applicability of IRM to bias mitigation---a special case of o.o.d generalization---in increasingly naturalistic settings and deep models. Using natural language inference (NLI) as a test case, we start with a setting where both the dataset and the bias are synthetic, continue with a natural dataset and synthetic bias, and end with a fully realistic setting with natural datasets and bias. Our results show that in naturalistic settings, learning complex features in place of the bias proves to be difficult, leading to a rather small improvement over empirical risk minimization. Moreover, we find that in addition to being sensitive to random seeds, the performance of IRM also depends on several critical factors, notably dataset size, bias prevalence, and bias strength, thus limiting IRM's advantage in practical scenarios. Our results  highlight key challenges in applying IRM to real-world scenarios, calling for a more naturalistic characterization of  the problem setup for o.o.d generalization. 

## Code structure
Each directory contains independent code to run a specific setting from the 3 settings described in the paper (toy experiment, synthetic bias and natural bias).
In each directory a sub-directory named "reproduce/experiments" contains the experiments in the paper for that setting and the run commands (in a "run_commands.txt" file) used to produce them. 
In the natrual bias setting there are two additional subdirectories: 
1. "reproduce/scores" which contains the run command used to train the biased model.
2. "scores" which contains the scores generated with the biased model. These scores are used to generate the environments in the natural bias settings.

All the code for the train and evaluation functions is found in "main.py" of the relevant setting. 

## Environment setup
Clone this repo:
```git clone https://github.com/technion-cs-nlp/irm-for-nli```

Then, from root folder generate and activate conda environment:
```
conda env create -f environment.yml
conda activate irm_for_nli
```

## Reproducing the results
As explained in the "code structure" section. 
Assume, for example, we want to reproduce the results for the bias prevalence analysis with hypothesis bias (experiment results described in Figure 1b in the paper). 
Then, to train the models run the file "natural_bias/reproduce/experiments/hypothesis_bias/snli/bias_prevalence_analysis/run_commands.txt".
To evaluate them run "natural_bias/reproduce/experiments/hypothesis_bias/snli/bias_prevalence_analysis/testing/run_commands.txt".
**Note:**  Before running, change the --outdir flag in the "run_commands.txt" files as you see fit.
