# PHSX815_Project4

This project runs two clustering algorithms, k-means clustering and gaussian mixture modeling, on exoplanet data in order to identify possibly astrophysically relevant subgroups of exoplanets.

PlotPlanets.py can be run to query exoplanet data from the Exoplanet Archive, save a csv of relevant exoplanet data, and save two text files, massperiod.txt and radperiod.txt containing JSON serialized mass-period and radius-period data. By manually editing the source file, PlotPlanets.py also overplots the fit k-means centers and gaussian mixture modeling ellipsoids. 

usage: python python/PlotPlanets.py

Cluster.py will read in a specified input file, and take a specified number of clusters, and run both k-means and gaussian mixture modeling on the input file data. It will also plot the results for the k-means and gaussian mixture modeling clusters.

usage: python python/Cluster.py -K [integer] -input [filename]

-K 		(optional) 	integer number of cluster >= 1. If not passed, default 2

-input 	(mandatory)	filename with stored JSON data in format specified from PlotPlanets

Required dependencies are: numpy, scipy, matplotlib, sklearn, pyvo