# Author: mycal@mit.edu
# Date: 06/28/2019

# Script for analyzing what observations map to what states.
# Saves images of the atari game in directories, organized by keys representing
# encodings or states. Further analysis of images could go at the bottom of this
# script.
import numpy as np
import os
import pickle
import scipy.misc

# Hardcoded parameter that points to the saved pkl files.
base_path = 'results/Atari/PongDeterministic-v4/gru_32_hx_(64,100)_bgru/'
# There are three pkl files of interest.
pkl_files = ['obs_to_encoding', 'obs_to_min_states', 'obs_to_unmin_states']
for relevant_pkl_file in pkl_files:
	# Load in the data from the specified paths.
	obs_to_encoding = {}
	with open(base_path + relevant_pkl_file + '.pkl', 'rb') as f:
		obs_to_encoding = pickle.load(f)

	# Flip the dictionary that uses observations as keys to instead use encodings as keys that
	# now point to the list of observations (conventiently reformatted into image format now).
	encoding_to_images = {}
	num_entries_to_check = 400
	num_entries_checked = 0
	for key, value in obs_to_encoding.items():
		# Reshape the key to 80x80, the format of the images
		observation_image = np.reshape(key, (80, 80))

		if value in encoding_to_images.keys():
			# Corresponds to multiple images for the same state
			encoding_to_images.get(value).append(observation_image)
		else:
			encoding_to_images[value] = [observation_image]

		num_entries_checked += 1
		if num_entries_checked >= num_entries_to_check:
			break

	# Save the images to disk.
	encoding_counter = 0
	for encoding, images in encoding_to_images.items():
		print("Saving images for encoding number ", encoding_counter)
		analysis_dir = base_path + 'analysis'
		type_of_mapping_dir = analysis_dir + '/' + relevant_pkl_file
		encoding_dir = type_of_mapping_dir + "/" + str(encoding_counter)
		try:
			if not os.path.exists(analysis_dir):
				os.mkdir(analysis_dir)
			if not os.path.exists(type_of_mapping_dir):
				os.mkdir(type_of_mapping_dir)
			if not os.path.exists(encoding_dir):
				os.mkdir(encoding_dir)
			else:
				print("WARNING: directory for images already exists. May overwrite existing data.")
		except OSError:
			print("Creating my directory failed.", encoding_dir)
		else:
			for i, image in enumerate(images):
				print("Saving image number", i)
				scipy.misc.imsave(encoding_dir + "/" + str(i) + '.jpg', image)
		encoding_counter += 1