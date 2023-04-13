# WSigKernel: Weighted Signature Kernels
General Signature Kernel Methods / Weighted Signature Kernel Methods

By Thomas Cass, Terry Lyons, and Xingcheng Xu

## Overview
Suppose that γ and σ are two continuous bounded variation paths which take values in a finite-dimensional inner product space V. Recent papers have respectively introduced the truncated and the untruncated signature kernel of γ and σ, and showed how these concepts can be used in classification and prediction tasks involving multivariate time series. In this paper, we introduce a general notion of the signature kernel, showing how these objects can be interpreted in many examples as an average of PDE solutions, and thus how they can be estimated computationally using suitable quadrature formulae. We extend this analysis to derive closed-form formulae for expressions involving the expected (Stratonovich) signature of Brownian motion. In doing so we articulate a novel connection between signature kernels and the notion of the hyperbolic development of a path, which has been a broadly useful tool in the recent analysis of the signature. As applications we evaluate the use of different general signature kernels as a basis for non-parametric goodness-of-fit tests to Wiener measure on path space.

The paper on our General/Weighted Signature Kernel Methods will be forthcoming on Annals of Applied Probability. The early version can be found on arXiv:

Thomas Cass, Terry Lyons, and Xingcheng Xu. “Weighted Signature Kernels.” Annals of Applied Probability (AAP), accepted/forthcoming (2023+). (Early access: as “General Signature Kernels” on arXiv preprint arXiv:2107.00447 (2021))

<a href="https://arxiv.org/abs/2107.00447" title="[arXiv Link]" target="_blank">[arXiv Link]</a> <a href="https://arxiv.org/pdf/2107.00447.pdf" title="[PDF on arXiv (the first version)]" target="_blank">[PDF on arXiv (the first version)]</a>

## Description

* The code "wsigkernel.py": for computing of weighted signature kernels, including the full original signature kernel, using PDE and quadrature where needed. 

* The code "hyperdevelop_explicit_factorial.py": for computing of the hyperbolic development and expected general signature kernels involving the expected (Stratonovich) signature of Brownian motion under the factorial weights. "hyperdevelop_explicit_beta.py" and "hyperdevelop_explicit_original.py" are for the Beta-weights and for the orginal case, respectively.

* The code: "optmeasure.py": for the computing of optimal discrete measures on paths using weighted signature kernels and the hyperbolic development.

* The codes in "examples" provide numerical results to illustrate the usefulness of general signature kernels in measuring the similarity or alignment between a given discrete measures on paths and Wiener measure, including Discrete Measures on Brownian Paths, Examples using cubature formulae, Applications in Signal Processing.

* The codes in "time_series_classfication" provide the use of weighted signature kernels applied to the challenge of multivariate time series classification using
the UEA datasets, which are available at https://timeseriesclassification.com/. We run the experiments both with and without augmenting the time series by adding an extra time coordinate. The code for augmenting is in "paths_transform.py".

## Code Reference
* https://github.com/crispitagorico/sigkernel

## Citation

Please cite the paper if you use the idea or code in this paper/repo.

```
@misc{WSigKernel,
  author = {Thomas Cass, Terry Lyons, and Xingcheng Xu},
  title = {Weighted Signature Kernels},
  year = {2023+},
  publisher = {Institute of Mathematical Statistics},
  journal = {Annals of Applied Probability},
}
```

Early Version on arXiv:
```
@misc{WSigKernel,
  author = {Thomas Cass, Terry Lyons, and Xingcheng Xu},
  title = {General Signature Kernels},
  year = {2021},
  publisher = {arXiv},
  journal = {arXiv preprint},
  howpublished = {\url{https://arxiv.org/abs/2107.00447}},
}
```

