# Here's a way to generate baseline analysis.
# Start with some number of clusters and some number of items. Randomly assign them.
import matplotlib
matplotlib.use('Agg')  # Needed by Mycal for not breaking remote display issues.
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment


def create_random_allocation(clusters_to_sizes):
	num_total_items = sum([val for key, val in clusters_to_sizes.items()])
	# print("Num total items", num_total_items)  # For debugging
	permuted_ints = np.random.permutation(num_total_items)
	int_idx = 0
	allocation = {}
	for key, num_items in clusters_to_sizes.items():
		items_for_key = []
		for _ in range(num_items):
			items_for_key.append(permuted_ints[int_idx])
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

def run_cluster_simulation(cluster_sizes):
	clusters1 = create_random_allocation(cluster_sizes)
	clusters2 = create_random_allocation(cluster_sizes)

	max_length = max(len(clusters1.keys()), len(clusters2.keys()))
	# Now create the overlap
	overlap = create_clustering_matrix(clusters1, clusters2, max_length)
	# Analyse the overlap
	sum_of_overlap = create_optimal_alignment(overlap, max_length)
	return sum_of_overlap

def run_baselines():
	cluster_sizes = {0: 1, 1: 2, 2: 3}  # 3 clusters of size 5 each
	run_1_cluster_size = {0 : 17, 1 : 1, 2 : 1, 3 : 167, 4 : 79, 5 : 20, 6 : 20, 7 : 15, 8 : 20, 9 : 18, 10 : 14, 11 : 1}

	num_values = sum([cluster for cluster in run_1_cluster_size.values()])

	num_simulations = 1000
	sim_results = np.zeros(num_simulations)
	for i in range(num_simulations):
		print("Running sim number", i)
		sum_for_sim = run_cluster_simulation(run_1_cluster_size)
		sim_results[i] = sum_for_sim

	# Now the simulations are all over: print statistics and show histogram.
	print(stats.describe(sim_results))
	hist, _ = np.histogram(sim_results, bins=np.arange(num_values))

	plt.plot(hist)
	plt.savefig('analysis/random_trials.png')
	plt.close()