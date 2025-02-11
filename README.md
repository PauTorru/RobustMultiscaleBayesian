Robust Multiscale Bayesian Energy Dispersive X-ray spectroscopy quantification
===============================================================================

This library contains a series of functions to perform quantification of EDX data using the RMB method in a streamlined manner. The scripts use the ESPM library. Basic understanding of its concepts is necessary to use the RMB method.

Usage
------
Maps, Variances =  get_RMB_maps(s,I_resol) 
Directly gives elemental maps and its pixel-wise variances, according to the multiscale strategy defined by I_resol.

The elemental signals can be quantified by:

Quantified_maps, Quantification_error = atomic_percentage_and_uncertainty_from_RMB_maps(Maps, Variances)

Follow the RMB_demo.ipynb notebook for further details

Requirements
------------
The following are needed:

  * https://github.com/Laadr/SUNSAL

  * https://github.com/adriente/espm


CITING
------

If you use this library, please cite on of the following :

