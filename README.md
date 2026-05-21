# [A dual-mode crystal structure prediction framework for fixed element sets and fixed structural prototypes]()

Here, we propose a deep-learning-enabled dual-mode crystal structure prediction framework that simultaneously supports two complementary tasks: 
-predicting stable crystal structures for given elemental compositions. 
-identifying chemically viable elemental substitutions for a predefined crystal topology.

##  Prerequisites

This package requires:

- [torch](https://pytorch.org/)

- [pymatgen](https://pymatgen.org/)

- [ase](https://ase-lib.org/) 

- [smact](https://smact.readthedocs.io/en/latest/smact.html) (optional: Required for composition screening using SMACT)

- [matgl](https://matgl.ai/) (optional: Required for relaxing structures using the general atomic potential)

## Usage
For detailed usage, please refer to prediction.ipynb

## Authors

This package was primarily written by Chenglong Qin (clqin@xhu.edu.cn.com).

## License

The model is released under the MIT License.