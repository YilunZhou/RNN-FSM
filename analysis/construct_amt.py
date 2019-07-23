# Script reads in a list of pairs of clusters (e.g. pong/run0/cluster0 and pong/run0/cluster1).
# It adds links to those images to a csv file in the right format to be uploaded to AMT
import csv
import numpy as np
import os

# Fix seed for reproducability.
np.random.seed(0)

# Parameters to set for the script.
min_num_obs_in_cluster = 5  # What's the minimum number of images a cluster must have to be included?
cluster_display_size = 4  # How many images to show for each cluster (probably want min_num_obs_in_cluster - 1)

website_prefix = "https://mycal-tucker.github.io/assets/"
website_game_name = "pong"
game = 'PongDeterministic-v4'
run_number = 0
csv_file_name = 'analysis/amt_' + website_game_name + str(run_number) + '.csv'


clusters = []

# In this case, generates the pairs of clusters programmatically by doing all pairs for a particular run.
run_path = "results/Atari/" + game + "-run" + str(run_number)
min_states_dir = run_path + "/gru_32_hx_(64,100)_bgru/analysis/obs_to_min_states"
for cluster0 in os.listdir(path=min_states_dir):
	cluster0_path = min_states_dir + "/" + cluster0
	if len(os.listdir(path=cluster0_path)) < min_num_obs_in_cluster:
		print("Skipping cluster with too few images ", cluster0)
		continue
	for cluster1 in os.listdir(path=min_states_dir):
		cluster1_path = min_states_dir + "/" + cluster1
		if len(os.listdir(path=cluster1_path)) < min_num_obs_in_cluster:
			continue
		if cluster0 == cluster1:
			continue
		# Note: this will create duplicates in the form of cluster0, cluster1 and cluster1, cluster0
		# That's not necessarily bad, as it randomizes what's left and right.
		else:
			clusters.append((cluster0_path, cluster0, cluster1_path, cluster1))

# Now, given the clusters, generate random subsets of the images from each cluster, and one image
# that will need to be assigned to one of the clusters.
amt_rows = []
for cluster0_path, cluster0_num, cluster1_path, cluster1_num in clusters:
	images_in_cluster0 = os.listdir(cluster0_path)
	images_in_cluster1 = os.listdir(cluster1_path)
	# Randomly choose a matching and distractor cluster.
	# TODO: if you wanted, could create multiple rows per cluster pairing.
	# It would be easy to do; it would just need an extra nested loop.
	matching_images = images_in_cluster0
	distractor_images = images_in_cluster1
	matching_cluster = cluster0_num
	distractor_cluster = cluster1_num
	if np.random.random() < 0.5:	
		matching_images = images_in_cluster1
		distractor_images = images_in_cluster0
		matching_cluster = cluster1_num
		distractor_cluster = cluster0_num

	# Choose the target image and matching images from matching cluster
	matching_subset = np.random.choice(matching_images, size=cluster_display_size + 1, replace=False)
	special_image = matching_subset[0]
	matching_subset = matching_subset[1:]
	distractor_subset = np.random.choice(distractor_images, size=cluster_display_size, replace=False)

	matching_prefix = website_prefix + website_game_name + str(run_number) + "/" + matching_cluster + "/"
	distractor_prefix = website_prefix + website_game_name + str(run_number) + "/" + distractor_cluster + "/"
	row = [matching_prefix + special_image]
	# Half the time put matching on the left; other half put on the right.
	left_prefix = matching_prefix
	left_subset = matching_subset
	right_prefix = distractor_prefix
	right_subset = distractor_subset
	if np.random.random() < 0.5:
		left_prefix = distractor_prefix
		left_subset = distractor_subset
		right_prefix = matching_prefix
		right_subset = matching_subset

	clusters = [left_prefix + left_subset[0],
	left_prefix + left_subset[1],
	right_prefix + right_subset[0],
	right_prefix + right_subset[1],
	left_prefix + left_subset[2],
	left_prefix + left_subset[3],
	right_prefix + right_subset[2],
	right_prefix + right_subset[3]]

	row.extend(clusters)
	amt_rows.append(row)

# Now write the amt_rows list into a csv file.
with open(csv_file_name, mode='w') as csv_file:
	writer = csv.writer(csv_file, delimiter=',')
	writer.writerow(['image_url', 'left_image_url1', 'left_image_url2', 'right_image_url1', 'right_image_url2', 'left_image_url3', 'left_image_url4', 'right_image_url3', 'right_image_url4'])
	for row in amt_rows:
		writer.writerow(row)