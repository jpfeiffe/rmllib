# Relational Machine Learning Library (RMLLib)

The Relational Machine Learning Library (rmllib) is aimed at providing scalable relational machine learning solutions in python.

## Features
* Collective inference for relational inference
* Semi-supervised learning utilizing esimates of labels for previous rounds
* Scalable solutions for single-box machines
* Additional implementations of state-of-the-art generative graph models for synthetic experimentation

## Getting started

The crux of rmllib focuses on a [relational dependency network](http://www.jmlr.org/papers/volume8/neville07a/neville07a.pdf) representation, where a set of *conditional* distributions (e.g, Relational Naive Bayes) of a label given its neighbors is laced together via a *collective inference* algorithms (e.g., Variational Inference).  On top of this, rmllib provides [semi-supervised learning and inference](https://jpfeiffe.github.io/pubs/WWW2015_MaxEntInf.pdf) methods that perform well in sparsely labeled data scenarios.

RMLLib uses APIs inspired by sklearn and relies heavily on numpy, scipy and pandas for data wrangling and optimizations, but generally these are not compatible learners for RMLLib.  This is largely due to the interconnectedness between labeled and unlabeled data.  The RMLLib dataformat largely hides this problem from the user by providing / using masking functions in the dataset to ensure the training labels remain unobserved during training.

For a simple example of building data and running methods, please see [the provided notebook](docs/notebooks/GettingStarted.ipynb).

## Installation
Currently, installation is only from source, i.e.:

> https://github.com/jpfeiffe/rmllib <br>
> cd rmllib <br>
> pip install rmllib

## Blame
Currently the project is maintained by me, Joel Pfeiffer, and until this project gets big feel free to [email me](mailto:jpfeiffe@gmail.com).  I'm always looking for help with new methods.  

If you find the library useful for your work, please consider citing:

> @misc{rmllib, <br>
>   title = {Relational Machine Learning Library (RMLLib)},<br>
>   author = {Joseph J. {Pfeiffer III}},<br>
>   howpublished = {\url{https://github.com/jpfeiffe/rmllib}},<br>
>   note = {Accessed: 2010-09-30}<br>
> }
