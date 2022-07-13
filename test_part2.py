import os
import sys


def test_part2():
	os.chdir('part2_denoise_cats')

	if not os.path.exists('cats_small.npy'):
		os.system('wget https://www.dropbox.com/s/acu1ygwzgk98l0h/cats_small.npy')

	from evaluate_cats import evaluate_on_dataset

	loss = evaluate_on_dataset('cats_small.npy')

	assert loss < 0.01


