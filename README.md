# Hypergraph-MT: mixed-membership community detection in hypergraphs

Python implementation of Hypergraph-MT algorithm described in:

[1] Authors (year). _title_, arXiv:.

If you use this code please cite this [article](website) (_preprint_).     

This is a probabilistic generative model that infers overlapping communities in hypergraphs. Thus, it is a mixed-membership model where we assume an assortative structure. The  inference is performed using an efficient expectation-maximization (EM) algorithm that exploits the sparsity of the network, leading to an efficient and scalable implementation.

Notice that when applied to graphs (considering only pairwise interactions), Hypergraph-MT reduces to [MultiTensor](https://github.com/cdebacco/MultiTensor) with assortative affinity matrix, as presented in [De Bacco et al. (2017)](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.95.042317). 

Copyright (c) 2022 [author list]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## What's included
- `code` : Contains the Python implementation of Hypergraph-MT algorithm, the code to run the inference and the hyperedge prediction task, and a Jupyter-notebook to show how to analyse the results.
- `data/input` : Contains the Workplace dataset used in the manuscript ([source](http://www.sociopatterns.org/datasets/contacts-in-a-workplace/)). 
- `data/output` : Contains some results.

## Requirements
The project has been developed using Python 3.7 with the packages contained in *requirements.txt*. We suggest to create a conda environment with
`conda create --name Hypergraph-MT python=3.7.9 --no-default-packages`, activate it with `conda activate Hypergraph-MT`, and install all the dependencies by running (inside `Hypergraph-MT` directory):

```bash
pip install -r requirements.txt
```

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
