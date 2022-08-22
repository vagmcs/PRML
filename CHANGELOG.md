## 0.2.0 (2022-08-22)

### Bug Fixes
- **chapter1:** Corrected variance of Gaussian over t ([3f60b33](https://github.com/vagmcs/prml/commit/3f60b3338550362db3928d8a2c9e04d31239c631))
- **chapter1:** Refactor to PolynomialBasis ([bfb6411](https://github.com/vagmcs/prml/commit/bfb64114e14bcf78badacbe40df4ab88526e2348))
- **chapter2:** Missing variables on mixtures of Gaussian ([ce3deb2](https://github.com/vagmcs/prml/commit/ce3deb2c8193a462cbc74eab75004939bf7661f4))
- **chapter2:** Removes redundant equality symbols ([44e1e31](https://github.com/vagmcs/prml/commit/44e1e315108e2b58b1dc111475bf78789ad5ad23))
- **chapter3:** Makes some figures more visible ([919ee08](https://github.com/vagmcs/prml/commit/919ee083e5af776c9fcd239e04538e81e330b7a5))
- **chapter3:** Bug fix in bayesian regression ([d57b907](https://github.com/vagmcs/prml/commit/d57b907056033f2d191c1b81a0769a52cf2c7c61))
- Changes var and c arguments to sigma ([db2f368](https://github.com/vagmcs/prml/commit/db2f368a3ba81ae99b16f18bcff36752fc147f83))
- Multivariate gaussian needs flatten mean when sampling ([05c86ff](https://github.com/vagmcs/prml/commit/05c86ffdd44f2b03fcef847b4183e4eeb489f7eb))

### Build
- Adds command for jupyter to makefile ([19cb217](https://github.com/vagmcs/prml/commit/19cb217c5978cd952bc3358fc3831c4d06fd74db))
- Flake8 checks and type hint fixes ([a914bc3](https://github.com/vagmcs/prml/commit/a914bc32c78b6ebb485efeda394805e347160909))
- Make command to generate PDF from notebooks ([8bf7bb5](https://github.com/vagmcs/prml/commit/8bf7bb5d2bed94595ee2787e1a25b3c03053660e))
- Poetry and make ([3e7a836](https://github.com/vagmcs/prml/commit/3e7a8367bb19f0399016cff064c6ef34a428959b))

### Docs
- Add articles for linear and ridge regression ([0983345](https://github.com/vagmcs/prml/commit/098334575065d6127947c3a9b4818d90c47b248b))
- Datasets methods types and docstrings ([b338aae](https://github.com/vagmcs/prml/commit/b338aae14560adb21ccd83ab7385adaf2d7716c6))
- Minor doc error on the Dirichlet ([8c40121](https://github.com/vagmcs/prml/commit/8c40121a02d0848d9193f627c6224ed9f80f90fb))

### Features
- **chapter 4:** Proofs for posterior probs on continuous inputs ([f4b5711](https://github.com/vagmcs/prml/commit/f4b57119843e20916b451e4a7148a9782a2c4df2))
- **chapter3:** Stochastic gradient descent ([56b87bd](https://github.com/vagmcs/prml/commit/56b87bd1c40f50741b632da2f1ca854109ec891c))
- **chapter4:** Adds Fisher's linear discriminant ([d661e2b](https://github.com/vagmcs/prml/commit/d661e2b2f40b152186620cecc23cf9ed0b593979))
- **chapter4:** Adds notebook ([403cb09](https://github.com/vagmcs/prml/commit/403cb09f34e621ca4d1775a274bf55b30059c828))
- **chapter4:** Adds perceptron algorithm ([d683200](https://github.com/vagmcs/prml/commit/d683200210ddcb5ced1acea73ff9f156d77e6d86))
- **chapter4:** Adds small section on gradient descent vs newton ([603b9f0](https://github.com/vagmcs/prml/commit/603b9f0cd4d8ff88a2ebda03c973f7783565cb44))
- **chapter4:** Bayesian logistic regression ([8adc374](https://github.com/vagmcs/prml/commit/8adc3742fe375658b30ed508c40731bce332ae6a))
- **chapter4:** Implements generative classifier ([210a35d](https://github.com/vagmcs/prml/commit/210a35d5a89881f97cb582c62e0e52ddf48bb482))
- **chapter4:** Implements softmax regression ([02ffdef](https://github.com/vagmcs/prml/commit/02ffdeffc05f04b080c64f5578f28a7f97b092fd))
- **chapter4:** Laplace approximation theory ([067eb0f](https://github.com/vagmcs/prml/commit/067eb0f7a3b982ed22edcfb2743fb5672eadc50d))
- **chapter4:** Least squares classifier ([93dd565](https://github.com/vagmcs/prml/commit/93dd565a74d963f52cc34f602c9d37d480a916f0))
- **chapter4:** Least squares classifier outliers example ([b1e796b](https://github.com/vagmcs/prml/commit/b1e796bfa8f6cf614e94531e68b5c62477bed50d))
- **chapter4:** Logistic regression intro ([e3c1523](https://github.com/vagmcs/prml/commit/e3c15232ddb38dcd79a15eb5a44ca1c15a06600d))
- **chapter4:** Logistic regression proofs and code ([5a4f7ec](https://github.com/vagmcs/prml/commit/5a4f7ecace78feaf7e0169b48f3cbf83298b620e))
- **chapter4:** Proofs for multiclass logistic regression ([bf9cf13](https://github.com/vagmcs/prml/commit/bf9cf135874ea6a0059caf290b430b7652d3cff2))

### Style
- **chapter4:** Applies code style ([5a5e180](https://github.com/vagmcs/prml/commit/5a5e180f137ec2a1effa00ab9e55b072b91649c9))
- **preprocessing:** Type and style fixes ([ca3fc83](https://github.com/vagmcs/prml/commit/ca3fc832331e3023355ff7ea11a8ad751563d23e))
- Enables higher resolution plots (retina) ([955897e](https://github.com/vagmcs/prml/commit/955897ea49e70d758451bb00b8527a6f54664a72))


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
