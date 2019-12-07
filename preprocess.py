import os
import pandas as pd
import sys
import numpy as np
from PIL import Image
import pickle as pkl

def pre_process(folder):
	files_in_folder = sorted(os.listdir(folder))
	df = pd.DataFrame()
	df['images'] = [np.array(Image.open(folder + x).resize((32, 32), Image.ANTIALIAS), dtype='uint8') for x in files_in_folder if 'color' in x]
	df['instance'] = [np.array(Image.open(folder + x).resize((32, 32), Image.ANTIALIAS), dtype='uint8') for x in files_in_folder if 'instanceIds' in x]
	df['label'] = [np.array(Image.open(folder + x).resize((32, 32), Image.ANTIALIAS), dtype='uint8') for x in files_in_folder if 'labelIds' in x]
	return df

pkl.dump(pre_process(sys.argv[1]), open('train.pkl', 'wb'))
pkl.dump(pre_process(sys.argv[2]), open('test.pkl', 'wb'))