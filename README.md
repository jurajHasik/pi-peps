# pi-peps

[![Build Status](https://travis-ci.com/jurajHasik/pi-peps.svg?branch=master)](https://travis-ci.com/jurajHasik/pi-peps)&nbsp;

C++ library built on top of ITensor for running iPEPS simulations of two dimensional spin systems. Wavefunctions are optimized through Simple Update or Full Update <cite>[1,2]</cite>. Expectation values and environments are computed by directional CTMRG algorithm <cite>[3]</cite>. 

<br>

For installation instruction and documentation continue to [pi-peps.readthedocs.io](https://pi-peps.readthedocs.io)

requirements:
* [ITensor](https://github.com/ITensor/ITensor) tensor algebra framework
* linear algebra: MKL or LAPACK & BLAS

optional:
* [ARPACK](https://github.com/opencollab/arpack-ng) iterative solver for large-scale (truncated)
matrix decompositions of both symmetric and non-symmetric matrices

included:
* [JSON for Modern C++](https://github.com/nlohmann/json) JSON library handling input and output files
* [RSVDPACK](https://github.com/sergeyvoronin/LowRankMatrixDecompositionCodes) approximate solver
for large-scale truncated singular value decompositions (requires MKL)

<br>
<br>

<cite>[1]</cite> H. C. Jiang, Z. Y. Weng, T. Xiang [Phys. Rev. Lett. 101, 090603 (2008)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.101.090603) or [arxiv:0806.3719](https://arxiv.org/abs/0806.3719)

<cite>[2]</cite> H. N. Phien, J. A. Bengua, H. D. Tuan, P. Corboz, R. Orus [Phys. Rev. B 92, 035142 (2015)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.035142) or [arxiv:1503.05345](https://arxiv.org/abs/1503.05345)

<cite>[3]</cite> P. Corboz, T. M. Rice, M. Troyer, [Phys. Rev. Lett. 113, 046402 (2014)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.113.046402) or [arxiv:1402.2859](https://arxiv.org/abs/1402.2859)

