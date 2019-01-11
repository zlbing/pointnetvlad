import pandas as pd
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import random

#####For training and test data split#####
x_width=20
y_width=20

def check_in_test_set(northing, easting, points, x_width, y_width):
	in_test_set=False
	for point in points:
		if(point[0]-x_width<northing and northing< point[0]+x_width and point[1]-y_width<easting and easting<point[1]+y_width):
			in_test_set=True
			break
	return in_test_set
##########################################

def output_to_file(output, filename):
	with open(filename, 'wb') as handle:
	    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print("Done ", filename)


def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename, p, output_name):
	database_trees=[]
	test_trees=[]
	for folder in folders:
		print(folder)
		df_database= pd.DataFrame(columns=['file','northing','easting'])
		df_test= pd.DataFrame(columns=['file','northing','easting'])
		df_locations_path= os.path.join(base_path,runs_folder,folder,filename)
		print("df_locations_path=",df_locations_path)
		df_locations= pd.read_csv(df_locations_path, sep=',')
		# df_locations['timestamp']=runs_folder+folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
		# df_locations=df_locations.rename(columns={'timestamp':'file'})
		for index, row in df_locations.iterrows():
			#entire business district is in the test set
			if(output_name=="business"):
				df_test=df_test.append(row, ignore_index=True)
			elif(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
				df_test=df_test.append(row, ignore_index=True)
			df_database=df_database.append(row, ignore_index=True)

		database_tree = KDTree(df_database[['northing','easting']])
		test_tree = KDTree(df_test[['northing','easting']])
		database_trees.append(database_tree)
		test_trees.append(test_tree)

	test_sets=[]
	database_sets=[]
	for folder in folders:
		database={}
		test={} 
		df_locations= pd.read_csv(os.path.join(base_path,runs_folder,folder,filename),sep=',')
		df_locations['timestamp']=runs_folder+folder+pointcloud_fols+df_locations['timestamp'].astype(str)+'.bin'
		df_locations=df_locations.rename(columns={'timestamp':'file'})
		for index,row in df_locations.iterrows():				
			#entire business district is in the test set
			if(output_name=="business"):
				test[len(test.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
			elif(check_in_test_set(row['northing'], row['easting'], p, x_width, y_width)):
				test[len(test.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
			database[len(database.keys())]={'query':row['file'],'northing':row['northing'],'easting':row['easting']}
		database_sets.append(database)
		test_sets.append(test)		

	for i in range(len(database_sets)):
		tree=database_trees[i]
		for j in range(len(test_sets)):
			if(i==j):
				continue
			for key in range(len(test_sets[j].keys())):
				coor=np.array([[test_sets[j][key]["northing"],test_sets[j][key]["easting"]]])
				index = tree.query_radius(coor, r=3)
				#indices of the positive matches in database i of each query (key) in test set j
				test_sets[j][key][i]=index[0].tolist()

	output_to_file(database_sets, output_name+'_evaluation_database.pickle')
	output_to_file(test_sets, output_name+'_evaluation_query.pickle')

#For kaicheng
p1=[0,0]
p2=[78.8541,27.9149]
p3=[-3.61771,92.393]
p4=[-29.8497,14.4575]
p_dict={"kaicheng":[p1,p2,p3,p4]}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path= "../../benchmark_datasets/"

#For Oxford
folders=[]
runs_folder = "kaicheng/"
all_folders=sorted(os.listdir(os.path.join(BASE_DIR,base_path,runs_folder)))
#index_list=[5,6,7,9,10,11,12,13,14,15,16,17,18,19,22,24,31,32,33,38,39,43,44]
index_list=[0,1,2,3]
print(len(index_list))
for index in index_list:
	folders.append(all_folders[index])

print(folders)
construct_query_and_database_sets(base_path, runs_folder, folders, "/pointcloud/", "pose.csv", p_dict["kaicheng"], "kaicheng")
