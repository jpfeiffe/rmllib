# Relational Machine Learning Library (RMLLib)

The Relational Machine Learning Library (rmllib) is aimed at providing scalable relational machine learning solutions in python.

## Features
* Collective inference for relational inference
* Semi-supervised learning utilizing esimates of labels for previous rounds
* Scalable solutions for single-box machines
* Additional implementations of state-of-the-art generative graph models for synthetic experimentation

## Getting started

RMLLib uses APIs inspired by sklearn and relies heavily on numpy, scipy and pandas for data wrangling and optimizations, but generally these are not compatible learners for RMLLib.  This is largely due to the interconnectedness between labeled and unlabeled data.  The RMLLib dataformat largely hides this problem from the user by providing / using masking functions in the dataset to ensure the training labels remain unobserved during training.

For a simple example of building data and running methods, please see [the provided notebook](docs/notebooks/GettingStarted.ipynb).

## Learning and Inference

The crux of RMLLib focuses on a [relational dependency network](http://www.jmlr.org/papers/volume8/neville07a/neville07a.pdf) representation, where a set of *conditional* distributions (e.g, Relational Naive Bayes) of a label given its neighbors is laced together via a *collective inference* algorithms (e.g., Variational Inference).  On top of this, RMLLib provides [semi-supervised learning and inference](https://jpfeiffe.github.io/pubs/WWW2015_MaxEntInf.pdf) methods that perform well in sparsely labeled data scenarios.

For the optimization step, RMLLib follows RDNs by maximizing the *pseudo*likelihood, allowing for faster optimization of the parameter space.  For inference, RMLLib diverges slightly 

## Data Format

RMLLib is intended to run from the ground up on large, potentially multi-class datasets.  To facilitate this, the generic dataset class that wraps four basic datastructures:

* labels: a pandas DataFrame with rows indicating sample labels and columns as a multiindex with level 0 being the "Y" label and class values being level 1
* features: either a pandas DataFrame or SparseDataFrame, with feature values being level=0 feature name and feature values being level=1.  Categorical features are assumed to have a one-hot-encoding representation allowing for simple slicing and sparse matrix multiplication (see [Boston Medians](rmllib/data/load/boston.py) for a simple example).
* edges: either dense or sparse matrix containing the weight values between nodes.

In addition, the dataset module provides helpers such as masks for defining a training/test split, and helpers for creating training sets that obscure unlabeled parts of the graph.


## Installation
Currently, installation is only from source, i.e.:

> git clone https://github.com/jpfeiffe/rmllib <br>
> cd rmllib <br>
> pip install rmllib

## Blame
Currently the project is maintained by me, [Joel Pfeiffer](mailto:jpfeiffe@gmail.com).  I'm always looking for help with new methods.  

If you find the library useful for your work, please consider citing:

> @misc{rmllib, <br>
>   title = {Relational Machine Learning Library (RMLLib)},<br>
>   author = {Joseph J. {Pfeiffer III}},<br>
>   howpublished = {\url{https://github.com/jpfeiffe/rmllib}},<br>
>   note = {Accessed: 2010-09-30}<br>
> }

Additionally, please ensure to cite relevant articles for the corresponding methods, algorithms and/or datasets.
