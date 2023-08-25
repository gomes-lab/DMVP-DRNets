# Deep Reasoning Network Implementation of a Deep Multivariate Probit Model (DMVP-DRNets) Modeling Code

## Main contributors

This code and data were prepared by:

- Di Chen (Cornell University, Dept. of Computer Science), dc874@cornell.edu
- Yiwei Bai (Cornell University, Dept. of Computer Science), yb263@cornell.edu
- Daniel Fink (Cornell University, Laboratory of Ornithology), daniel.fink@cornell.edu
- Carla P. Gomes (Cornell University, Dept. of Computer Science), gomes@cs.cornell.edu

## Citation

This work accompanies the following publication:

- Courtney L. Davis, Yiwei Bai, Di Chen, Orin Robinson, Viviana R. Gutierrez, Carla P. Gomes, and Daniel Fink. Deep learning with citizen science data enables estimation of species diversity and composition at continental extents. Accepted at Ecology.

Please see the **LICENSE** file for additional terms of use, and for information on acquiring data for other purposes.

## Requirements

- Python 3.7
- PyTorch 0.4.1
- TensorFlow 1.8
- Torchvision 0.4.0
- Scikit-image
- Scikit-learn 0.19.2
- R 3.6.1
- Hmsc R library (see https://github.com/hmsc-r/HMSC)

Helper scripts also assume a Bourne or Bash shell, however they are short and can easily be converted to other platforms or entered manually.

## Description
More information about each application and dataset is included in the readme.md files in each subdirectory. The scripts and instructions for replicating the results from each experiment in the publication are in these subdirectories:

- **BBS** : experiments using the 2011 Breeding Bird Survey (BBS) dataset
- **eBird_entire** : experiments using the entire eBird dataset to examine the ecology utility
- **eBird_subsets** : experiments using the subsets of eBird dataset with different scales 
- **Norberg2019** : experiments using the 5 species datasets from **Norbeg et. al 2019**
