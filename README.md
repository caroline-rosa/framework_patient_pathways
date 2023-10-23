# Framework for assessing multi-perspective patient pathways

This project implements the framework proposed in https://doi.org/10.48550/arXiv.2309.14208 for assessing multi-perspective patient pathways.

The file *create_and_modify_MAG_patient_pathways.py* contains functions to create a MultiAspect Graph (MAG) to represent patient pathways from a CSV file and to manipulate it.

The file *dissimilarity.py* contains the dissimilarity function proposed in the article.

The file *plot_mag.py* contains a function to plot a  MAG that represents patient pathways.

The file *MAG.py* is part of the MAG module developed by Juliana Z. G. Mascarenhas, Klaus Wehmuth and Artur Ziviani. For more information, please refer to https://doi.org/10.1016/j.tcs.2016.08.017 and  https://doi.org/10.1093/comnet/cnaa042.

The notebook *Generate synthetic patient pathways.ipynb* generates a synthetic dataset of patient pathways with the necessary fields to test the framework. Its goal is not to provide a reliable dataset to foster healthcare assessments but to supply those who would like to test the framework but do not have access to a healthcare dataset with an event log with the necessary information.

The notebook *Framework.ipynb* is an example of how to assess multi-perspective patient pathways, and the file *dashboard.py* generates a dashboard to facilitate their analysis.