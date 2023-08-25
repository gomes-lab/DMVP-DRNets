# Deep Reasoning Network Implementation of a Deep Multivariate Probit Model (DMVP-DRNets) Modeling Code and Data

## Main contributors

This code and data were prepared by:

- Di Chen (Cornell University, Dept. of Computer Science), dc874@cornell.edu
- Yiwei Bai (Cornell University, Dept. of Computer Science), yb263@cornell.edu
- Daniel Fink (Cornell University, Laboratory of Ornithology), daniel.fink@cornell.edu
- Carla P. Gomes (Cornell University, Dept. of Computer Science), gomes@cs.cornell.edu

## Citation

This work accompanies the following publication:

- Courtney L. Davis, Yiwei Bai, Di Chen, Orin Robinson, Viviana R. Gutierrez, Carla P. Gomes, and Daniel Fink. Deep learning with citizen science data enables estimation of species diversity and composition at continental extents.

These data files are provided only for reviewing this paper and verifying the results. Please see the **LICENSE** file for additional terms of use, and for information on acquiring data for other purposes.

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


## eBird data description

Two data files are provided in their own zip (due to file size): **eBird_data.csv** and **map_inference_data.csv**.

## eBird_data.csv

This contains species counts and environmental data. The species count data come from eBird, the global citizen-science bird monitoring platform. We used a subset of data in which the time, date, and location of each survey were reported and observers recorded the number of individuals of all bird species detected and identified during the survey period, resulting in a "complete checklist" of species on the survey. The checklists used here were restricted to those collected with the "stationary" or "traveling" protocols from January 1, 2004 to February 2, 2019 within the spatial extent between 170 to 60 W longitude and between 20 to 60 latitude. All surveys were further restricted to those with durations of at most 1 hour and for traveling surveys at most 1km. For each checklist we extracted species counts for the 500 most frequently detected species within the study extent. The environmental data were joined to the species count data based on location and date information. More information about the data fields can be found in the Supplemental information.

Each line consists of 576 fields. The first row of the file consists of the name of each field, and starting from the second row, each row represents one data point or checklist. The first 4 fields are indices used for data processing and *cannot* be used as predictive features. Fields 5-76 are predictive features associated with each checklist. These features include descriptions of the search event and the environmental features of the neighborhood where the search took place. The last 500 fields are the counts of each species. Species counts reported as "-1" should be removed.  This data file is used to train the model and do the model selection.

Note: 3 fields from each checklist are redacted for privacy reasons.

## map_inference_data.csv

This is used to map species distributions. This file includes all of the environmental variables across a uniform 3km grid of locations that span much of North America.  We use the 64 fields that correspond to fields 13-76 in **eBird_data.csv**.
