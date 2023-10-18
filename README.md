# Framework for assessing multi-perspective patient pathways

This project provides the implementation of the framework proposed in https://doi.org/10.48550/arXiv.2309.14208 for assessing multi-perspective patient pathways.

The file *create_and_modify_MAG_patient_pathways.py* contains functions to create a MultiAspect Graph (MAG) to represent patient pathways from a csv file and to manipulate it.

The file *dissimilarity.py* contains the dissimilarity function proposed in the article.

The file *plot_mag.py* contains a funtion to plot a  MAG that represents patient pathways.

The file *MAG.py* is part of the MAG module implemented by Juliana Z. G. Mascarenhas, Klaus Wehmuth and Artur Ziviani. 

The notebook *Generate synthetic patient pathways.ipynb* generates a synthetic dataset of patient pathways with the necessary fields to test the framework. Its goal is not to provide a reliable set of data to foster healthcare assessments, but only to supply those who would like to test the framework but do not have access to a healthcare dataset at the moment, an eventlog with the necessary information to do so.

The notebook *Framework.ipynb* is an example of how to assess multi-perspective patient pathways and the file *dashboard.py* generates a dashboard to facilitate their analysis.