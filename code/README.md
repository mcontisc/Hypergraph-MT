# Hypergraph-MT: Python code
Copyright (c) 2022 [Martina Contisciani](https://www.is.mpg.de/person/mcontisciani) and [Caterina De Bacco](http://cdebacco.com).

Implements the algorithm described in:

[1] Contisciani M., Battiston F., and De Bacco C. (2022). _Principled inference of hyperedges and overlapping communities in hypergraphs_, arXiv:
2204.05646.

If you use this code please cite this [article](https://arxiv.org/abs/2204.05646) (_preprint_).     

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the 'Software'), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Files
- `HyMT` : Class definition of Hypergraph-MT, the algorithm to perform community detection in hypergraphs. 
- `main.py` : General version of the algorithm. It performs the inference in the given dataset, and it infers the latent variables as community memberships and affinity matrix. It also gives the option to run two baselines: run the model on the graph obtained by clique expansions (Graph-MT), and run the model on the graph given by the subset of pairwise interactions (Pairs-MT).
- `main_cv.py` : Code to perform a k-fold cross-validation procedure, for the hyperedge prediction task. It returns a *csv* file summarizing the AUC over all folds. 
- `tools.py` : Contains non-class functions for handling the data and analysing the results.
- `cv_functions.py` : Contains non-class functions for the cross-validation routine.
- `setting_workplace.yaml` : Setting to run the algorithm on the given dataset (input for *main.py* and *main\_cv.py*).
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
- **-U** : Flag to normalize u such that every row sums to 1, *(default=0)*

You can find a list by running (inside `code` directory): 

```bash
python main.py --help
```

## Input format
The network should be stored in a *.npz* file, containing:

- **A** : Array of length E, containing the weights of the hyperedges
- **B** : Incidence matrix of dimension N x E
- **hye** : Array of length E, containing the sets of hyperedges (as tuples)

where N is the number of nodes, and E is the number of hyperedges.
## Output
The algorithm returns a compressed file inside the `data/output` folder. To load and print the membership matrix:

```bash
import numpy as np 
theta = np.load('theta_file_name.npz')
print(theta['u'])
```

_theta_ contains the N x K-dimensional membership matrix **u** *('u')*, the D x K-dimensional affinity matrix **w** *('w')*, the total number of iterations *('max_it')*, the value of the maximum log-likelihood *('maxL')*, and the list of non-isolated nodes *('non_isolates')*.  
