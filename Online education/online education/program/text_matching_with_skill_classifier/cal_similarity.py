"""
calculate the similarity of different texts using key words
distance methods include: cosine, jaccard, euclidean (not effective)
"""

import pandas as pd
import numpy as np
import math
import copy
# import jieba
from utils import find_max_index
from utils import skill_classifier
from nltk.tokenize import word_tokenize


def cosine(v1,v2):
    v1_arr=np.array(v1)
    v2_arr=np.array(v2)
    up = np.sum(v1_arr*v2_arr)
    downl = np.power(np.sum(v1_arr*v1_arr),0.5)
    downr = np.power(np.sum(v2_arr * v2_arr), 0.5)
    cosine_=up/(downl*downr)
    return cosine_

# do not use count vector
def Jaccard(v1,v2):
    v1=set(v1)
    v2=set(v2)
    up=v1.intersection(v2)
    down=v1.union(v2)
    jaccard=1.0*len(up)/len(down)
    return jaccard

# use count vector
def Distance(v1,v2):
    v1_arr=np.array(v1)
    v2_arr=np.array(v2)
    distance=np.linalg.norm(v1_arr-v2_arr)
    return distance

def cal_similarity(bowA, bowB, type_ = 'cosine'):
    list_ = [bowA, bowB]
    # print(list_)

    # construct word set
    word_set = set(bowA).union(set(bowB))

    # construct index
    word_index_dict = {}
    for index, word in enumerate(word_set):
        word_index_dict[word]=index
    
    # count vector
    count_vector = []
    for text in list_:
        vector_list=[0]*len(word_set)
        for word in text:
            vector_list[word_index_dict[word]]+=1
        count_vector.append(vector_list)

    if(type_== 'cosine'):
        result = cosine(count_vector[0],count_vector[1])
        if math.isnan(result):
            result = 0
        return result
    elif(type_ == 'euclidean'):
        return Distance(count_vector[0],count_vector[1])
    else:
        print("lack para")


with open('../../skill_classifier/all_soft_skills.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]
soft_skills = []
for line in lines:
    words = word_tokenize(line.lower())
    soft_skills.extend(words)


# read job
file_path = '../../data_key/keywords job(ver2).xls'   
raw_data = pd.read_excel(file_path, header=0)  
job_data = raw_data.values

# read program
program_file = '../../data_key/keywords_program(2).xlsx'   
program_raw_data = pd.read_excel(program_file, header=0)  
program_data = program_raw_data.values
program_row = program_data[0][1:]
print(len(program_data))

# to get job and program name
# read job
df1 = pd.read_csv("../../data_pre/job data_pre(ver2).csv")
job_title = df1['job title']

# read program
df2 = pd.read_csv("../../data_pre/program data_pre(ver2).csv")
school_name = df2['Schoole']
program_name = df2['program']

df3 = pd.read_csv("../../eval/clustered_data.csv") 
cluster_data = df3["cluster"].values

    



rmd_result = []
cosine_score = 0
jaccard_score = 0
euclidean_score = 0
best_score = 0
worst_score = 0
for test_id in range(0,len(job_data)):
    row = job_data[test_id][1:]
    newlist = [x for x in job_data[test_id][1:] if pd.isnull(x) == False]
    print(newlist)

    cosine_list = []
    jaccard_list = []
    euclidean_list = []
    for program in program_data:
        program = program[1:]
        program = [x for x in program if pd.isnull(x) == False]
        cosine_list.append(cal_similarity(newlist, program[1:]))
        jaccard_list.append(Jaccard(newlist, program[1:]))
        euclidean_list.append(cal_similarity(newlist, program[1:], type_='euclidean'))
        # print("cosine:",cal_similarity(newlist, program[1:]),end='\t')
        # print('Jaccard:',Jaccard(newlist, program[1:]), end='\t')
        # print('Euclidean distance:',cal_similarity(newlist, program[1:], type_='euclidean'))

    # print(cosine_list)
    cosine_max_number, cosine_max_index = find_max_index(cosine_list, 3)
    jaccard_max_number, jaccard_max_index = find_max_index(jaccard_list, 3)
    euclidean_max_number, euclidean_max_index = find_max_index(euclidean_list, 3)
    # print(cosine_max_number)
    # print(cosine_max_index)

   
    df3 = pd.read_csv("../skill_classifier/clustered_data.csv") 
    cluster_data = df3["cluster"].values

    recom = [job_title[test_id]]
    print(job_title[test_id])
    print("matched programs:",end=' ')
    for item in euclidean_max_index:
        print("university name:",school_name[item],end=' ')
        print("program name:", program_name[item])
        recom.append(school_name[item] + '/' + program_name[item])
  
    cosine_score += 3 - len(set(cosine_max_index))
    jaccard_score += 3 - len(set(jaccard_max_index))
    euclidean_score += 3 - len(set(euclidean_max_index))
    worst_score += 3


    rmd_result.append(recom)

# print(rmd_result)
print(cosine_score)
print(jaccard_score)
print(euclidean_score)
print(best_score, worst_score)

import csv
header = ['job title', 'recommand1', 'recommand2', 'recommand3']

with open('recommaned_program_euclidean.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rmd_result)

