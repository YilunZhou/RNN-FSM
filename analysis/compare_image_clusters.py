# Script that analyzes how similar image clusters from two different runs are.
# TODO: what's a good metric?
# As a proof-of-concept, just match up exact image matches.
# import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os


PATH_DIVIDER = "/"
# Here are a few hard-coded paths that you can switch between.
run0 = "results/Atari/PongDeterministic-v4-run0/gru_32_hx_(64,100)_bgru/analysis/obs_to_min_states"
run1 = "results/Atari/PongDeterministic-v4-run1/gru_32_hx_(64,100)_bgru/analysis/obs_to_min_states"
run2 = "results/Atari/PongDeterministic-v4-run2/gru_32_hx_(64,100)_bgru/analysis/obs_to_min_states"
# Set the following 2 to point to the directories you care about.
# As a sanity check for the code, you can always set run1_dir to the same as run2_dir.
# You should see perfect overlap.
run1_dir = run1
run2_dir = run0

run1_data = {}
run2_data = {}
runs = [(run1_dir, run1_data), (run2_dir, run2_data)]


# Print out the passed-in dictionary, sorted by key. Highly brittle (assumes ints, prints lengths, etc.)
def print_sorted_dict(dict_to_print):
	for key, value in sorted(dict_to_print.items(), key=lambda x: int(x[0])):
		print("{} : {}".format(key, len(value)))

# Get all the clusters from run1 and all the clusters from run2.
for directory, data in runs:
	# print("Loading data from ", directory)
	for cluster in os.listdir(directory):
		cluster_path = directory + PATH_DIVIDER + cluster
		data[cluster] = []
		for image_name in os.listdir(cluster_path):
			image_path = cluster_path + PATH_DIVIDER + image_name
			image = plt.imread(image_path)
			data.get(cluster).append(image)
	# Debug line to manually sanity check that all the images are being read.
	# print_sorted_dict(data)

# At this point, have the data we want loaded into run1_data and run2_data, so need to compare.
# First, build a 2d array that shows the overlap between the clusters.
# Row value is the cluster for run1
# Column value is the cluster for run2
# Value is how many exact matches there are.
cluster_overlap = np.zeros([len(run1_data.keys()), len(run2_data.keys())])

for cluster1, images1 in run1_data.items():
	for cluster2, images2 in run2_data.items():
		exact_match_count = 0
		for image1 in images1:
			for image2 in images2:
				# If every single pixel is the same, counts as a match.
				if np.all(image1 == image2):
					exact_match_count += 1
					break
		cluster_overlap[int(cluster1)][int(cluster2)] = exact_match_count
print(cluster_overlap)

# Now can do more complex analysis, like what is the optimal alignment and how much overlap is there?
