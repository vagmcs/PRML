## 0.1.0 (2021-04-03)

### Features

- **chapter3**: Evidence approximation proofs finalized
- **dataset**: Datasets package implementation
- **chapter3**: Evidence approximation (cont)
- **linear**: Evidence approximation implementation
- **chapter3**: Evidence likelihood function proof
- **chapter3**: Evidence approximation basic theory
- **chapter3**: Bayesian regression example figures
- **chapter3**: Predictive distribution theory
- **linear**: Bayesian linear regression
- **chapter3**: Examples for bayesian linear regression
- **chapter3**: Parameter distribution proofs
- **chapter3**: Bias-variance decomposition
- **linear**: Gradient descent implementation
- **chapter3**: Examples of basis functions
- **feature**: Gaussian and sigmoid basis function
- **chapter3**: Notes on least squares and maximum likelihood
- **chapter3**: Chapter for linear regression
- **chapter2**: Notes on NN classification
- **chapter2**: Plots on k-nearest neighbor classifier
- **chapter2**: Notes on Gaussian kernel and kNN estimation
- **chapter2**: Notes on kernel density estimators
- **chapter2**: Notes on non-parametric density modelling
- **chapter2**: Notes on mixtures of Gaussian
- **datasets**: Old faithful dataset
- **chapter2**: Bayesian inference for the Gaussian
- **distribution**: Gamma and Student's t-distribution
- **chapter2**: Notes on Robbins-Monro algorithm
- **chapter2**: Sequential estimation of ML
- **chapter1**: Figures on curve fitting re-visited
- **chapter1**: Likelihood plots and proofs for unbiased variance
- **chapter1**: Notes updates and images on probability theory
- **distribution**: Multivariate Gaussian maximum likelihood
- **distribution**: Maximum likelihood for the multivariate Gaussian
- **chapter2**: Notes on the Dirichlet Bayesian inference
- **chapter2**: Bayesian inference example on multinomial variables
- **chapter2**: Example plots for the multinomial distribution
- **distribution**: Multinomial distribution implementation
- **chapter2**: Notes on the Dirichlet distribution
- **chapter2**: Plots on multinomial random variables
- **distribution**: Categorical distribution implementation
- **chapter1**: Conditional entropy theory and proofs
- **chapter1**: Proof of mutual information, entropy relation
- **chapter2**: Notes on multinomial random variables
- **chapter1**: Table of contents
- **chapter2**: Table of contents
- **chapter2**: Plots and proofs on binary random variables
- **distribution**: Symbolic distribution representation
- **distribution**: Beta and Dirichlet distribution implementations
- **chapter2**: Notes and examples on binary variables
- **chapter2**: Notebook for chapter 2
- **chapter1**: Multivariate Gaussian distribution
- **chapter1**: Information theory notes, proofs and plots
- **chapter1**: Notes on curve fitting from a Bayesian perspective
- **chapter1**: Curve fitting re-visited
- **chapter1**: Gaussian distribution ML notes, proofs and plots
- **chapter1**: Notes on Gaussian distribution
- **chapter1**: Notes, proofs on PDFs, expected value and variance
- **chapter1**: Sum, product rules and the Bayes theorem
- **chapter1**: Added ridge regression examples and plots
- **regression**: Ridge regression implementation

### Refactor

- **preprocessing**: Package feature renamed to preprocessing
- **feature**: Basis functions refactor and docs
- **distribution**: Refactored probability distribution implementations
- **distribution**: Dirichlet distribution code refactor

### Fixes

- **distribution**: Multivariate Gaussian needs flat mean when sampling
- **chapter1**: Rename to PolynomialBasis
- **chapter2**: Missing variables on mixtures of Gaussian
- **chapter1**: Corrected variance of the Gaussian
- **chapter2**: Removed LaTeX *equation environment*
- **chapter2**: Gaussian example plot bug and some typos
- **distribution**: Use `sym.Matrix` for the mean of the Gaussian
- **distribution**: Remove array wrapper of `np.power` in multinomial distribution
- **distribution**: Generic distribution pdf over D-dimensional variables
- **distribution**: Allow Gaussian integer parameters
- **distribution**: `MatrixSymbol` bug on the `change_notation` method
- **distribution**: Categorical maximum likelihood solution `np.mean` requires `axis=1`
- **distribution**: Fixed `self.D` initialization on categorical distribution
- **distribution**: Replaced `np.prod` in likelihood by `np.product`
- **chapter1**: Changed math mode single `$` to double `$$`
- **chapter1**: Rendering bug fixed in maximum likelihood solution for the variance
- **chapter1**: Duplicate text removed
- **chapter1**: Remove numbering from LaTeX *align environment*
- **chapter1**: Simple probability examples and minor math fixes
- **chapter1**: Fixed broken link to youtube video (3blue1brown)
