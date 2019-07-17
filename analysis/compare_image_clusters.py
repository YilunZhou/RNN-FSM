# Script that analyzes how similar image clusters from two different runs are.
# TODO: what's a good metric?
# As a proof-of-concept, just match up exact image matches.
from clustering_utils import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from scipy.optimize import linear_sum_assignment


PATH_DIVIDER = "/"
# Here are a few hard-coded paths that you can switch between.
run0 = "results/Atari/PongDeterministic-v4-run0/gru_32_hx_(64,100)_bgru/analysis/obs_to_min_states"
run1 = "results/Atari/PongDeterministic-v4-run1/gru_32_hx_(64,100)_bgru/analysis/obs_to_min_states"
run2 = "results/Atari/PongDeterministic-v4-run2/gru_32_hx_(64,100)_bgru/analysis/obs_to_min_states"
# Set the following 2 to point to the directories you care about.
# As a sanity check for the code, you can always set run1_dir to the same as run2_dir.
# You should see perfect overlap.
run1_dir = run1
run2_dir = run2

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
# Count how many images there are for each run.
num_run1_images = sum([len(cluster) for cluster in run1_data.values()])
num_run2_images = sum([len(cluster) for cluster in run2_data.values()])

# At this point, have the data we want loaded into run1_data and run2_data, so need to compare.
max_length = max(len(run1_data.keys()), len(run2_data.keys()))
cluster_overlap = create_clustering_matrix(run1_data, run2_data, max_length)
print(cluster_overlap)

# Now can do more complex analysis, like what is the optimal alignment and how much overlap is there?
# Use the hungarian algorithm, which minimizes the sum of the permutation.
# Subtract from max value so that we get values that we want to minimize.
max_value = max([max(row) for row in cluster_overlap])
unmatched_values = max_value * np.ones([max_length, max_length]) - cluster_overlap
# print(unmatched_values)

sum_of_matches = create_optimal_alignment(cluster_overlap, max_length, print_alignment=True)

# Print out the key results.
print("Run1 count", num_run1_images)
print("Run2 count", num_run2_images)
print("Number of matches across runs", sum_of_matches)

# As a baseline, just count how many of the images from different runs are the same, ignoring
# the actual clusters they're put into.
verify_no_repeats(run1_data)
verify_no_repeats(run2_data)
max_overlap = count_maximal_overlap(run1_data, run2_data)
print("Max overlap", max_overlap)
