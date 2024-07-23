# WSigKernel: Weighted Signature Kernels
General Signature Kernels / Weighted Signature Kernels

By Thomas Cass, Terry Lyons, and Xingcheng Xu

## Overview
Suppose that γ and σ are two continuous bounded variation paths which take values in a finite-dimensional inner product space V. Recent papers have respectively introduced the truncated and the untruncated signature kernel of γ and σ, and showed how these concepts can be used in classification and prediction tasks involving multivariate time series. In this paper, we introduce a general notion of the signature kernel, showing how these objects can be interpreted in many examples as an average of PDE solutions, and thus how they can be estimated computationally using suitable quadrature formulae. We extend this analysis to derive closed-form formulae for expressions involving the expected (Stratonovich) signature of Brownian motion. In doing so we articulate a novel connection between signature kernels and the notion of the hyperbolic development of a path, which has been a broadly useful tool in the recent analysis of the signature. As applications we evaluate the use of different general signature kernels as a basis for non-parametric goodness-of-fit tests to Wiener measure on path space.

The paper on our General/Weighted Signature Kernels is published in Annals of Applied Probability:

```
Thomas Cass, Terry Lyons, and Xingcheng Xu. “Weighted Signature Kernels.” Annals of Applied Probability,
Vol. 34, No. 1A, (2024), 585-626. (Early access: as “General Signature Kernels” on arXiv preprint arXiv:2107.00447 (2021))
```

<a href="https://arxiv.org/abs/2107.00447" title="[arXiv Link]" target="_blank">[arXiv Link]</a> <a href="https://projecteuclid.org/journals/annals-of-applied-probability/volume-34/issue-1A/Weighted-signature-kernels/10.1214/23-AAP1973.short" title="[Journal Version]" target="_blank">[Journal Version]</a> <a href="https://xingchengxu.github.io/Publications/WSK_CLX2024_AAP.pdf" title="[PDF]" target="_blank">[PDF]</a> <a href="https://github.com/xingchengxu/WSigKernel" title="[GitHub Code]" target="_blank">[GitHub Code]</a>

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
@article{CLX2024,
  title={Weighted signature kernels},
  author={Cass, Thomas and Lyons, Terry and Xu, Xingcheng},
  journal={The Annals of Applied Probability},
  volume={34},
  number={1A},
  pages={585--626},
  year={2024},
  publisher={Institute of Mathematical Statistics}
}
```

Early Version on arXiv:
```
@article{CLX2021,
  title={General signature kernels},
  author={Cass, Thomas and Lyons, Terry and Xu, Xingcheng},
  journal={arXiv preprint arXiv:2107.00447},
  year={2021}
}
```

