import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import pyvo as vo
import json


# connect to exoplanet archive and grab requested data
service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
results = service.search("select discoverymethod, st_spectype, pl_bmassj, pl_orbper, hostname, pl_radj from pscomppars ", maxrec=6000)
table = results.to_table()

# save results to csv
table.to_pandas().to_csv("planets.csv")

methods = []
sptyps = []
ms = []
pers = []
names = []
rads = []


# read all data into usable arrays
for line in table:
    methods.append(line[0])
    sptyps.append(line[1])
    ms.append(line[2])
    pers.append(line[3])
    names.append(line[4].strip())
    rads.append(line[5])

# get the unique methods and names for plotting later
unique_methods = np.array(list(set(methods)))
unique_names = set(names)

print(unique_methods)
print(len(sptyps))
print(len(unique_names))
print(len(rads))

# save mass-period data, radius-period data to json and print to files
massperiod = {
    "x": [],
    "y": [],
    "meth": [],
}

radperiod = {
    "x": [],
    "y": [],
    "meth": [],
}

for i in range(len(methods)):
    if (np.isnan(float(pers[i])) or np.isnan(float(rads[i]))):
        pass
    else:
        radperiod["x"].append(float(pers[i]))
        radperiod["y"].append(float(rads[i]))
        radperiod["meth"].append(str(methods[i]))
        
    if (np.isnan(float(pers[i])) or np.isnan(float(ms[i]))):
        pass
    else:
        massperiod["x"].append(float(pers[i]))
        massperiod["y"].append(float(ms[i]))
        massperiod["meth"].append(str(methods[i]))
        
massperiod["x"] = np.log10(massperiod["x"]).tolist()
massperiod["y"] = np.log10(massperiod["y"]).tolist()
        
outfile = open("massperiod.txt", 'w')
outfile.write(json.dumps(massperiod))
outfile.close()

radperiod["x"] = np.log10(radperiod["x"]).tolist()

outfile = open("radperiod.txt", 'w')
outfile.write(json.dumps(radperiod))
outfile.close()



# convert possible text nans to np.nans
def FloatOrNaN(value):
    try:
        return float(value)
    except:
        print("nan")
        return np.nan

ms = np.array(list(map(FloatOrNaN, ms)))
pers = np.array(list(map(FloatOrNaN, pers)))
rads = np.array(list(map(FloatOrNaN, rads)))


print(len(ms), len(pers), len(rads))

# keep track of all the planets associated with each method
traces = {
    'Radial Velocity': [[],[],[]],
    'Transit': [[],[],[]],
    'Imaging': [[],[],[]],
    'Microlensing': [[],[],[]],
    'Transit Timing Variations': [[],[],[]],
    'Other': [[],[],[]],
}

for i in range(len(ms)):
    if methods[i] in traces.keys():
        traces[methods[i]][0].append(ms[i])
        traces[methods[i]][1].append(pers[i])
        traces[methods[i]][2].append(rads[i])
    else:
        traces["Other"][0].append(ms[i])
        traces["Other"][1].append(pers[i])
        traces["Other"][2].append(rads[i])


# plot discovery method data for both mass-period and radius-period distributions
fig, ax = plt.subplots(1, 2, figsize=(20, 8))

plt.rcParams.update({'font.size': 22})
for i, trace in enumerate(traces):
    clr = "C{}".format(i)
    print(trace)
    if trace == "Other":
        clr = "k"
    ax[0].plot(traces[trace][1], traces[trace][0], ".", color=clr, markersize=7.5, rasterized=True)
    ax[1].plot(traces[trace][1], traces[trace][2], ".", color=clr, markersize=7.5, rasterized=True)

    
ax[0].set_xscale("log")
ax[1].set_xscale("log")

ax[0].set_yscale("log")

plt.rcParams.update({'font.size': 16})

ax[1].legend(traces.keys())

plt.rcParams.update({'font.size': 18})
ax[0].set_title("Mass-Period Distribution")
ax[1].set_title("Radius-Period Distribution")

ax[0].set_ylabel(r"Planet Mass ($M_{Jup}$)")
ax[1].set_ylabel(r"Planet Radius ($R_{Jup}$)")


ax[0].set_xlabel(r"Orbit Period (days)")
ax[1].set_xlabel(r"Orbit Period (days)")
plt.show()
fig.savefig("plots/exo_distributions.jpg", dpi=200)




# manually adding k-means clustering, gmm clustering data to the plots

# collecting clustering results - mass/period

k_clusters = np.array([
    [ 0.66328766, -0.01388814],
    [ 1.05602402, -1.8160057 ],
    [ 2.90628407,  0.30247658]])

gmm_clusters = np.array([
    [0.6067256895249435, -0.07623793234424127],
    [1.0720978097991372, -1.8111947736412568],
    [2.8347784507785727, 0.33207763744569785],
])

gmm_sigmas = np.array([
    [0.849133422059847, 1.620311317615842],
    [0.9610563596605202, 1.6872729988646138],
    [1.5184155302393645, 1.8861235819228443],
])

gmm_angles = [
    9.104646920665129,
    -60.372615080174526,
    -62.84767274913065,
]

# plot data again

fig, ax = plt.subplots(1, 2, figsize=(20, 8), sharey=False)

plt.rcParams.update({'font.size': 22})
for i, trace in enumerate(traces):
    clr = "C{}".format(i)
    print(trace)
    if trace == "Other":
        clr = "k"
    ax[0].plot(np.log10(traces[trace][1]), np.log10(traces[trace][0]), ".", color=clr, markersize=7.5, rasterized=True)
    
    
for i in range(len(gmm_angles)):
    ell = mpl.patches.Ellipse(gmm_clusters[i], gmm_sigmas[i][0], gmm_sigmas[i][1], 180.+gmm_angles[i], facecolor="none", lw=2, edgecolor="k", zorder=10000)
    #ell.set_alpha(0.4)
    ax[0].add_artist(ell)
ax[0].plot(k_clusters[:,0], k_clusters[:,1], '*w', markersize=20, mec="k", mew=2, zorder=100000)

plt.rcParams.update({'font.size': 16})
ax[0].legend(traces.keys())
plt.rcParams.update({'font.size': 18})
ax[0].set_title("Mass-Period Cluster Data with Discovery Method")

ax[0].set_ylabel(r"log10 Mass ($M_{Jup}$)")
ax[0].set_xlabel(r"log10 Period (days)")

# collecting clustering results - radius/period

k_clusters1 = np.array([
    [0.54267886, 0.51717801],
    [1.37098195, 0.26436091],
    [2.91814335, 1.09737705]])

gmm_clusters1 = np.array([
    [2.8349004255431396, 1.1773400653646975],
    [1.0239398533106556, 0.18318049295178485],
    [0.5499215064182731, 1.2220817569144913],
    [1.50445228187471, 0.487374695326412]
])

gmm_sigmas1 = np.array([
    [0.1425113065403681, 1.8524362241247008],
    [0.16817887590684852, 1.4523318764882924],
    [0.6072747755617383, 0.8365866346864654],
    [0.6885247915543548, 2.263229433570608]
])

gmm_angles1 = [
    89.12189349029371,
    -86.30361087077851,
    33.842794399224914,
    -81.37524545257513
]

plt.rcParams.update({'font.size': 22})
for i, trace in enumerate(traces):
    clr = "C{}".format(i)
    print(trace)
    if trace == "Other":
        clr = "k"
    ax[1].plot(np.log10(traces[trace][1]), np.array(traces[trace][2]), ".", color=clr, markersize=7.5, rasterized=True)
    
    
for i in range(len(gmm_angles1)):
    ell = mpl.patches.Ellipse(gmm_clusters1[i], gmm_sigmas1[i][0], gmm_sigmas1[i][1], 180.+gmm_angles1[i], facecolor="none", lw=2, edgecolor="k", zorder=10000)
    #ell.set_alpha(0.4)
    ax[1].add_artist(ell)
ax[1].plot(k_clusters1[:,0], k_clusters1[:,1], '*w', markersize=20, mec="k", mew=2, zorder=100000)

plt.rcParams.update({'font.size': 16})
ax[1].legend(traces.keys())
plt.rcParams.update({'font.size': 18})
ax[1].set_title("Radius-Period Cluster Data with Discovery Method")

ax[1].set_ylabel(r"log10 Radius ($R_{Jup}$)")
ax[1].set_xlabel(r"log10 Period (days)")
plt.show()
fig.savefig("plots/cluster_discmeth.jpg", dpi=300)
