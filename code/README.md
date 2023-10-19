# Hypergraph-MT: Python code
Copyright (c) 2022 [Martina Contisciani](https://martinacontisciani.wixsite.com/mcontisciani) and [Caterina De Bacco](http://cdebacco.com).

Implementation of the algorithm described in:

[1] Contisciani M., Battiston F., and De Bacco C. (2022). [_Inference of hyperedges and overlapping communities in hypergraphs_](https://rdcu.be/c0qdd), Nature Communications, 13:7229, 2022.

If you use this code please cite [[1]](https://www.nature.com/articles/s41467-022-34714-7#citeas). Details can be found in the [_published version_](https://doi.org/10.1038/s41467-022-34714-7) or in the [_preprint_](https://arxiv.org/abs/2204.05646).    

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the 'Software'), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Files
- `HyMT` : Folder that contains the implementation of Hypergraph-MT (`model.py`), and other utils files. 
- `main.py` : General version of the algorithm. It performs the inference in the given dataset, and it infers the latent variables as community memberships and affinity matrix. It also gives the option to run two baselines: run the model on the graph obtained by clique expansions (Graph-MT), and run the model on the graph given by the subset of pairwise interactions (Pairs-MT).
- `main_cv.py` : Code to perform a k-fold cross-validation procedure, for the hyperedge prediction task. It returns a *csv* file summarizing the AUC over all folds. 
- `tools.py` : Contains non-class functions for handling the data and analysing the results.
- `cv_functions.py` : Contains non-class functions for the cross-validation routine.
- `vizCM.py` : Contains a class object for visualizing and handling communities.
- `setting_workplace.yaml` : Setting to run the algorithm on the given dataset (input for `main.py` and `main\_cv.py`).
- `analyse_results.ipynb` : Example Jupyter-notebook to import the output results.

## Usage
To test the program on the given example file, type:  

```bash
cd code
python main.py
```

It will use the Workplace dataset contained in `data/input`. 

See the demo [jupyter notebook](https://github.com/mcontisc/Hypergraph-MT/blob/main/code/analyse_results.ipynb) for an example on how to analyse the output results, including visualizing the inferred communities.

### Parameters
- **-f** : Path of the input folder, *(default='../data/input/')*
- **-d** : Name of the dataset to analyse, *(default='workplace')*
- **-K** : Number of communities, *(default=5)*
- **-v** : Flag to print details, *(default=1)*
- **-b** : Flag to run the baselines, *(default=1)*
- **-D** : Threshold for the highest degree (size hyperedge) to keep, *(default=None')*

You can find this list by running (inside `code` directory): 

```bash
python main.py --help
```

### Setting file
In addition to the listed parameters, `main.py` and `main_cv.py` take in input a _yaml_ file. This `setting_<dataset>.yaml` file contains additional parameters to pass to the model, that are:
- **seed** : Number to set the seed of the random number generator
- **constraintU** : Flag to normalize the community matrix $u$ such that every row sums to 1. 
- **fix_communities** : Flag to set the communities $u$ as the input file and fix them during the inference.
- **fix_w** : Flag to set the affinity matrix $w$ as the input file and fix it during the inference.
- **gammaU** : Constant to regularize the communities $u$.
- **gammaW** : Constant to regularize the affinity matrix $w$.
- **initialize_u** : Option to initialize the communities $u$ different from random.
- **initialize_w**: Option to initialize the affinity matrix $w$ different from random.
- **out_inference** : Flag to output the inference results.
- **out_folder** : Path of the output folder.
- **end_file** : Suffix of the output file.
- **plot_loglik** : Flag to plot the log-likelihood.

## Input format
The network should be stored in a *.npz* file, containing:

- **A** : Array of length $E$, containing the weights of the hyperedges
- **B** : Incidence matrix of dimension $N \times E$
- **hye** : Array of length $E$, containing the sets of hyperedges (as tuples)

where $N$ is the number of nodes, and $E$ is the number of hyperedges.

## Output
The algorithm returns a compressed file inside the `data/output` folder. To load and print the membership matrix:

```bash
import numpy as np 
theta = np.load('theta_file_name.npz')
print(theta['u'])
```

**theta** contains the $N \times K$-dimensional membership matrix $u$ ('u'), the $D \times K$-dimensional affinity matrix $w$ ('w'), the total number of iterations ('max_it'), the value of the maximum log-likelihood ('maxL'), and the list of non-isolated nodes ('non_isolates').  

When you run `main.py` with the parameter `baselines=1` and `out_inference=True`, then the algorithm will output three files, corresponding to the three different models:
- **\_HyMT** is the one by using Hypergraph-MT
- **\_GrMT** is the one by using Graph-MT
- **\_PaMT** is the one by using Pairs-MT
