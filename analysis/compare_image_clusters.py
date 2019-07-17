from __future__ import division
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
all_runs = [run0, run1, run2]


# Print out the passed-in dictionary, sorted by key. Highly brittle (assumes ints, prints lengths, etc.)
def print_sorted_dict(dict_to_print):
	for key, value in sorted(dict_to_print.items(), key=lambda x: int(x[0])):
		print("{} : {}".format(key, len(value)))

# Create a class that will bundle all the metrics we want for comparing runs.
class Metrics:
	def __init__(self, num_run1_images, num_run2_images, num_possible_matches, num_clustered_matches,
				raw_baseline_stats, total_baseline_stats, common_baseline_stats):
		self.num_run1_images = num_run1_images
		self.num_run2_images = num_run2_images
		self.num_possible_matches = num_possible_matches
		self.num_clustered_matches = num_clustered_matches
		self.raw_baseline_stats = raw_baseline_stats
		self.total_baseline_stats = total_baseline_stats
		self.common_baseline_stats = common_baseline_stats

		self.min_run_images = min([num_run1_images, num_run2_images])
		self.max_run_images = max([num_run1_images, num_run2_images])

	def get_percent_common_total(self, relative_to_min=True):
		relevant_denominator = self.min_run_images
		if not relative_to_min:
			relevant_denominator = self.max_run_images
		return self.num_possible_matches / relevant_denominator

	def get_percent_clustered_matches_total(self, relative_to_min=True):
		relevant_denominator = self.min_run_images
		if not relative_to_min:
			relevant_denominator = self.max_run_images
		return self.num_clustered_matches / relevant_denominator

	def get_percent_clustered_matches_from_common(self):
		return self.num_clustered_matches / self.num_possible_matches

	def get_raw_baseline_stats(self):
		return (self.raw_baseline_stats.mean, self.raw_baseline_stats.variance)
	def get_total_baseline_stats(self):
		return (self.total_baseline_stats.mean, self.total_baseline_stats.variance)
	def get_common_baseline_stats(self):
		return (self.common_baseline_stats.mean, self.common_baseline_stats.variance)


# Create a 2D array to track the metrics for all pairwise combinations of runs
combination_metrics = []
for i, run1_dir in enumerate(all_runs):
	metrics_from_run1 = []
	for j, run2_dir in enumerate(all_runs):
		if j > i:
			continue  # No need to do the upper-right half of the matrix
		print("Analysing runs " + str(i) + " and " + str(j))

		run1_data = {}
		run2_data = {}
		runs = [(run1_dir, run1_data), (run2_dir, run2_data)]


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
		# print(cluster_overlap)

		# Now can do more complex analysis, like what is the optimal alignment and how much overlap is there?
		# Use the hungarian algorithm, which minimizes the sum of the permutation.
		# Subtract from max value so that we get values that we want to minimize.
		max_value = max([max(row) for row in cluster_overlap])
		unmatched_values = max_value * np.ones([max_length, max_length]) - cluster_overlap
		# print(unmatched_values)

		sum_of_matches = create_optimal_alignment(cluster_overlap, max_length, print_alignment=False)

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
		raw_baseline, raw_percent_total, raw_percent_common = run_baselines(run1_data, run2_data, max_overlap,
			file_suffix=str(i) + "_" + str(j))

		metric_for_run = Metrics(num_run1_images, num_run2_images, max_overlap, sum_of_matches,
			raw_baseline, raw_percent_total, raw_percent_common)
		metrics_from_run1.append(metric_for_run)
	combination_metrics.append(metrics_from_run1)

# Now break up the data in combination_metrics into interesting tables
num_runs = len(combination_metrics)
percent_common_total = np.zeros(shape=(num_runs, num_runs))
percent_clustered_total = np.zeros(shape=(num_runs, num_runs))
percent_clustered_common = np.zeros(shape=(num_runs, num_runs))
raw_baseline_stats = np.zeros(shape=(num_runs, num_runs, 2))
total_baseline_stats = np.zeros(shape=(num_runs, num_runs, 2))
common_baseline_stats = np.zeros(shape=(num_runs, num_runs, 2))
all_matrices = [percent_common_total, percent_clustered_total, percent_clustered_common,
raw_baseline_stats, total_baseline_stats, common_baseline_stats]
for i, row in enumerate(combination_metrics):
	for j, metric in enumerate(row):
		if j > i:
			continue
		percent_common_total[i][j] = metric.get_percent_common_total()
		percent_clustered_total[i][j] = metric.get_percent_clustered_matches_total()
		percent_clustered_common[i][j] = metric.get_percent_clustered_matches_from_common()
		raw_baseline_stats[i][j][0], raw_baseline_stats[i][j][1] = metric.get_raw_baseline_stats()
		total_baseline_stats[i][j][0], total_baseline_stats[i][j][1] = metric.get_total_baseline_stats()
		common_baseline_stats[i][j][0], common_baseline_stats[i][j][1] = metric.get_common_baseline_stats()
# Reflect around the diagonal to fill in the empty entries.
for i in range(num_runs):
	for j in range(num_runs):
		if j > i:
			for matrix in all_matrices:
				matrix[i][j] = matrix[j][i]

print("Percent of images between runs in common, ignoring clusters")
print(percent_common_total)
print()
print("Percent of images that can be aligned while respecting clusters")
print(percent_clustered_total)
print()
print("Percent of images that can be aligned while respecting clusters, compared to all common images")
print(percent_clustered_common)
print()
print("Random baseline statistics for counting number of clustered aligned images.")
for baseline_stats in [raw_baseline_stats, total_baseline_stats, common_baseline_stats]:
	pretty_baseline_stats = []
	for row in baseline_stats:
		new_row = []
		for col in row:
			mean = col[0]
			var = col[1]
			# Right now, leave out variance
			new_row.append(str(mean)) # + " (" + str(var) + ")")
		pretty_baseline_stats.append(new_row)
	# print(baseline_stats)
	print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in pretty_baseline_stats]))
	print()