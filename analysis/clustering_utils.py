from __future__ import division
# Here's a way to generate baseline analysis.
# Start with some number of clusters and some number of items. Randomly assign them.
import matplotlib
matplotlib.use('Agg')  # Needed by Mycal for not breaking remote display issues.
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment


def create_random_allocation(clusters_to_sizes,  num_small_ints=1000, upper_range_offset=1000):
	num_total_items = sum([val for key, val in clusters_to_sizes.items()])
	# print("Num total items", num_total_items)  # For debugging
	permuted_ints = np.random.permutation(num_total_items)
	int_idx = 0
	allocation = {}
	for key, num_items in clusters_to_sizes.items():
		items_for_key = []
		for _ in range(num_items):
			value_to_assign = permuted_ints[int_idx]
			if int_idx >= num_small_ints:
				value_to_assign += upper_range_offset
			items_for_key.append(value_to_assign)
			int_idx += 1
		allocation[key] = items_for_key
	return allocation

def create_clustering_matrix(clusters1, clusters2, max_length):
	cluster_overlap = np.zeros([max_length, max_length])
	for cluster1, values1 in clusters1.items():
		for cluster2, values2 in clusters2.items():
			exact_match_count = 0
			for value1 in values1:
				for value2 in values2:
					# If every single pixel is the same, counts as a match.
					if np.all(value1 == value2):
						exact_match_count += 1
						break
			cluster_overlap[int(cluster1)][int(cluster2)] = exact_match_count
	return cluster_overlap

def create_optimal_alignment(cluster_overlap, max_length, print_alignment=False):
	# Now can do more complex analysis, like what is the optimal alignment and how much overlap is there?
	# Use the hungarian algorithm, which minimizes the sum of the permutation.
	# Subtract from max value so that we get values that we want to minimize.
	max_value = max([max(row) for row in cluster_overlap])
	unmatched_values = max_value * np.ones([max_length, max_length]) - cluster_overlap
	# print(unmatched_values)

	row_ind, col_ind = linear_sum_assignment(unmatched_values)
	if print_alignment:
		print(row_ind)
		print(col_ind)
	# Now, use those rows and columns to sum up how many actually match.
	sum_of_matches = 0
	for i, row_idx in enumerate(row_ind):
		col_idx = col_ind[i]
		sum_of_matches += cluster_overlap[row_idx][col_idx]
	return sum_of_matches

def verify_no_repeats(cluster):
	list_of_values = []
	for values1 in cluster.values():
		for value in values1:
			list_of_values.append(value)
	num_values = len(list_of_values)
	num_duplicates = 0
	for value in list_of_values:
		seen_repeat = False
		for val2 in list_of_values:
			if np.all(val2 == value):
				if seen_repeat:
					num_duplicates += 1
				else:
					seen_repeat = True  # Allowed one repeat for self
	if num_duplicates != 0:
		print("PANIC! There were " + str(num_duplicates) + " duplicates in " + str(num_values) + " total")
	# Can add back in else statement if want to debug more.
	# else:
	# 	print("No duplicates found")

def count_maximal_overlap(clusters1, clusters2):
	sum_of_matches = 0
	for values1 in clusters1.values():
		for values2 in clusters2.values():
			for value1 in values1:
				for value2 in values2:
					# If every single pixel is the same, counts as a match.
					if np.all(value1 == value2):
						sum_of_matches += 1
	return sum_of_matches

# Take in two dictionaries mapping cluster ids to number of images in each cluster.
# Also accept a parameter that says how many of the ints in there should be shared
# at all. This sort of represents when two policies diverge and therefore have no
# chance of overlapping again.
def run_cluster_simulation(cluster_sizes1, cluster_sizes2, num_vals_common):
	clusters1 = create_random_allocation(cluster_sizes1)
	clusters2 = create_random_allocation(cluster_sizes2, num_small_ints=num_vals_common)

	max_length = max(len(clusters1.keys()), len(clusters2.keys()))
	# Now create the overlap
	overlap = create_clustering_matrix(clusters1, clusters2, max_length)
	# Analyse the overlap
	sum_of_overlap = create_optimal_alignment(overlap, max_length)
	return sum_of_overlap

def run_baselines(clusters1, clusters2, num_vals_common, file_suffix=''):
	clusters1_sizes = {}
	clusters2_sizes = {}
	for key, val in clusters1.items():
		clusters1_sizes[int(key)] = len(val)
	for key, val in clusters2.items():
		clusters2_sizes[int(key)] = len(val)

	num_values1 = sum([cluster for cluster in clusters1_sizes.values()])
	num_values2 = sum([cluster for cluster in clusters2_sizes.values()])

	num_simulations = 100
	raw_count_results = np.zeros(num_simulations)
	percent_match_total_results = np.zeros(num_simulations)
	percent_match_common_results = np.zeros(num_simulations)
	for i in range(num_simulations):
		print("Running sim number", i)
		sum_for_sim = run_cluster_simulation(clusters1_sizes, clusters2_sizes, num_vals_common)
		raw_count_results[i] = sum_for_sim
		percent_match_total_results[i] = sum_for_sim / min([num_values1, num_values2])
		percent_match_common_results[i] = sum_for_sim / num_vals_common

	# Now the simulations are all over: print statistics and show histogram.
	raw_count_analysis = stats.describe(raw_count_results)
	percent_match_total_analysis = stats.describe(percent_match_total_results)
	percent_match_common_analysis = stats.describe(percent_match_common_results)
	raw_hist, _ = np.histogram(raw_count_results, bins=np.arange(min([num_values1, num_values2])))
	percent_bins = np.arange(start=0, stop=1.05, step=0.05)
	total_hist, _ = np.histogram(percent_match_total_results, bins=percent_bins)
	common_hist, _ = np.histogram(percent_match_common_results, bins=percent_bins)
	plt.plot(percent_bins[:-1], common_hist)
	plt.savefig('analysis/common_hist_' + file_suffix + '.png')
	plt.close()
	return raw_count_analysis, percent_match_total_analysis, percent_match_common_analysis