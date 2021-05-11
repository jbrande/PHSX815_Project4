# #! /usr/bin/env python

# imports of external packages to use in our code
import sys
import numpy as np
import re
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import mixture
import itertools
from scipy import linalg

# main function for our coin toss Python code
if __name__ == "__main__":
	# if no args passed (need at least the input file), dump the help message
	# if the user includes the flag -h or --help print the options
	if '-h' in sys.argv or '--help' in sys.argv or len(sys.argv) == 1:
		print ("Usage: %s [-input0 string] [-input1 string] [-Ndice int]" % sys.argv[0])
		print ("-input:      (mandatory) the name of the file which holds the sample data")
		print ("-K:	 		 (optional) the number of clusters to attempt to find in the data, default 2. must be >= 1")
		print
		sys.exit(1)


	# number of clusters to try and find
	K = 2

	maxiter = 100 # maximum number of iterations to do before quitting

	# read the user-provided arguments from the command line (if there)
	if '-input' in sys.argv:
		p = sys.argv.index('-input')
		try:
			input0 = sys.argv[p+1]
		except IndexError as e:
			print("Must pass input filename")
	else:
		print("Must pass input filename")
	if '-K' in sys.argv:
		p = sys.argv.index('-K')
		K = int(sys.argv[p+1])
		if K < 1:
			print("K must be >= 1")
			sys.exit(1)

	# load json data from file
	res0 = ""
	with open(input0) as f:
		res0 = json.load(f)
		f.close()
	true_mus = np.array(res0["meth"])

	x = []
	y = []

	gmm_data = []

	for i in range(len(res0["x"])):
		if np.isnan(res0["x"][i]) or np.isnan(res0["y"][i]):
			pass
		else:
			x.append(res0["x"][i])
			y.append(res0["y"][i])
			gmm_data.append(np.array([res0["x"][i], res0["y"][i]]))

	x = np.array(x)#np.log10(np.array(x))#
	y = np.array(y)#np.log10(np.array(y))#

	gmm_data = np.array(gmm_data)

	# get the number of points
	num_points = len(x)

	print("Running k-means clustering")
	# keep track of how many iterations we're doing
	iteration = 0

	# need initial domain of data
	dmn = [[np.min(x), np.max(x)],[np.min(y), np.max(y)]]
	#print(dmn)

	# get K initial uniformly random guesses
	means = np.random.uniform([dmn[0][0], dmn[1][0]], [dmn[0][1], dmn[1][1]], (K,2))

	#print(means)

	# keep track of last guess to compare to, something outside of domain at first
	last = np.random.uniform([dmn[0][1], dmn[1][1]], [dmn[0][1] + 1, dmn[1][1] + 1], (K,2))

	# iteratively do K-means
	while iteration < maxiter:
		dists = []

		# find distances of all points to all guessed means
		for k in range(K):
			dist = np.sqrt((x - means[k][0])**2 + (y - means[k][1])**2)
			dists.append(np.array(dist))
		dists = np.array(dists)


		# keep track of the clusters 
		clusters = [ [[], []] for _ in range(K)] # K empty arrays to hold clusters

		# associate each point with its closest mean
		for i in range(num_points):
			ind = np.argmin(dists[:,i])
			clusters[ind][0].append(x[i])
			clusters[ind][1].append(y[i])


		# if any of the clusters are empty, quit. this happens sometimes when there are "too many" means for the number of points, or one guess is really bad and is far away
		# not sure how to handle this, so just die
		for k in range(K):
			if len(clusters[k][0]) == 0:
				print("empty cluster")
				sys.exit(1)

		# find centroids of clusters to see where to step to next
		centroids = []
		for k in range(K):
			cx = np.mean(clusters[k][0])
			cy = np.mean(clusters[k][1])
			centroids.append(np.array([cx, cy]))
		centroids = np.array(centroids)

		# check if new centroids are in the same place as the old means
		cdists = []
		for k in range(K):
			cdists.append(np.sqrt((centroids[k][0] - means[k][0])**2 + (centroids[k][1] - means[k][1])**2))

		# if the new cluster centroids and the old means are the same, stop iterating because we're done
		if np.array_equal(cdists, np.zeros(K)):
			break;

		# swap centers around for next iteration
		last = means
		means = centroids

		# increment iteration
		iteration = iteration + 1

	# print the final centers:
	print("The final k-means cluster centers are:")
	print(means)

	# make dual plot to handle both final results
	fig, ax = plt.subplots(1, 2, sharey=False, figsize=(12, 5))

	# plot k-means side
	for k in range(K):
		ax[0].plot(clusters[k][0], clusters[k][1], ".", markersize=2, c="C{}".format(k))
		ax[0].plot(means[k][0], means[k][1], "*", markersize=10, mec="k", c="C{}".format(k))

	ax[0].set_xlabel("log10 Period (days)")
	ax[0].set_ylabel(r"log10 Mass ($M_{Jup}$)")
	#ax[0].set_ylabel(r"log10 Radius ($R_{Jup}$)")
	ax[0].set_title("Exoplanet K-means Results")

	# run gaussian mixture modeling, following the sklearn example: https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm.html#sphx-glr-auto-examples-mixture-plot-gmm-py
	# in this case, just do normal expectation maximization
	color_iter = itertools.cycle(['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
	def plot_results(X, Y_, means, covariances, index, title):
		# loop through each cluster
		print("The Gaussian Mixture clusters are:")
		for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
			# get semiaxes and rotation angle from fit gaussian ellipsoids
			v, w = linalg.eigh(covar)
			v = 2. * np.sqrt(2.) * np.sqrt(v)
			u = w[0] / linalg.norm(w[0])

			# as the DP will not use every component it has access to
			# unless it needs it, we shouldn't plot the redundant
			# components.
			if not np.any(Y_ == i):
				continue
			ax[1].scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

			# Plot an ellipse to show the Gaussian component
			angle = np.arctan(u[1] / u[0])
			angle = 180. * angle / np.pi  # convert to degrees
			ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
			ax[1].plot(mean[0], mean[1], "*", markersize=10, mec="k", c=color)
			print(i)
			print([mean[0], mean[1]])
			print([v[0], v[1]])
			print(angle)

			ell.set_alpha(0.5)
			ax[1].add_artist(ell)

		ax[1].set_xlabel("log10 Period (days)")
		#ax[1].set_ylabel(r"log10 Radius ($R_{Jup}$)")
		ax[1].set_ylabel("log10 Mass (Mjup)")
		ax[1].set_title(title)



	print("Running Gaussian Mixture Modeling - Expectation Maximization ")
	gmm = mixture.GaussianMixture(n_components=K, covariance_type="full").fit(gmm_data)

	plot_results(gmm_data, gmm.predict(gmm_data), gmm.means_, gmm.covariances_, 0, 'Exoplanet Gaussian Mixture')
	fig.savefig("plots/mp_dual_plot.jpg", dpi=200)
	#fig.savefig("plots/rp_dual_plot.jpg", dpi=200)
	plt.show()
	plt.close()
