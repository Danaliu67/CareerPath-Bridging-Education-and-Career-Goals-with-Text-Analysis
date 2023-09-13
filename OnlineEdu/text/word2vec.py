import numpy as np
import pandas as pd
# import copy

from gensim.models import Word2Vec
from utils import find_max_index


def map_keywords_to_vectors(keyword_list, model):
    keyword_vectors = []
    for keyword in keyword_list:
        if keyword in model.wv.key_to_index:
            keyword_vectors.append(model.wv[keyword])
    return keyword_vectors



def calc_keyword_similarity(keyword_list_1, keyword_list_2, model):
    keyword_vectors_1 = map_keywords_to_vectors(keyword_list_1, model)
    keyword_vectors_2 = map_keywords_to_vectors(keyword_list_2, model)
    len_ = min(len(keyword_vectors_1), len(keyword_vectors_2))
    if(len_==0):
        return 0
    else:
        if(len(keyword_vectors_1) > len_):
            keyword_vectors_1 = keyword_vectors_1[:len_-len(keyword_vectors_1)]
        if(len(keyword_vectors_2) > len_):
            keyword_vectors_2 = keyword_vectors_2[:len_-len(keyword_vectors_2)]
        keyword_vectors_1 = np.array(keyword_vectors_1)
        keyword_vectors_2 = np.array(keyword_vectors_2)
        sim = np.dot(keyword_vectors_1, keyword_vectors_2.T) / \
            (np.linalg.norm(keyword_vectors_1, axis=1) * np.linalg.norm(keyword_vectors_2, axis=1))
        sim_mean = np.mean(sim)
        return sim_mean


file_path = 'keywords job(ver2).xls'   
raw_data = pd.read_excel(file_path, header=0)  
job_data = raw_data.values

# read program
program_file = 'keywords_program(2).xlsx'   
program_raw_data = pd.read_excel(program_file, header=0)  
program_data = program_raw_data.values


df1 = pd.read_csv("job data_pre(ver2).csv")
job_title = df1['job title']

# read program
df2 = pd.read_csv("program data_pre(ver2).csv")
school_name = df2['Schoole']
program_name = df2['program']

df3 = pd.read_csv("../../../eval/eval/clustered_data.csv") 
cluster_data = df3["cluster"].values
model = Word2Vec.load("../checkpoints/word2vec_model.bin")


cosine_score = 0
jaccard_score = 0
euclidean_score = 0
best_score = 0
worst_score = 0

similarity_mat=np.zeros((len(job_data),len(program_data)))

for test_id in range(0,len(job_data)):
    # print(job_data[test_id])
    row = job_data[test_id][1:]
    # newlist = [x if pd.isnull(x) == False else '<pad>' for x in row]
    newlist = [x for x in row if pd.isnull(x) == False]
    # print(newlist)

    cosine_list = []
    jaccard_list = []
    euclidean_list = []
    for program_id in range(len(program_data)):
        program = program_data[program_id][1:]
        program = [x for x in program if pd.isnull(x) == False]
        similarity=calc_keyword_similarity(newlist, program, model)
        cosine_list.append(similarity)
        similarity_mat[test_id,program_id]=similarity


rmd_program_result=[]
rmd_job_result=[]
for test_id in range(len(job_data)):
    cosine_list=list(similarity_mat[test_id,:])
    cosine_max_number, cosine_max_index = find_max_index(cosine_list, 3)
    recom_program = [job_title[test_id]]
    for item in cosine_max_index:
        recom_program.append(school_name[item] + '/' + program_name[item])
    rmd_program_result.append(recom_program)
for program_id in range(len(program_data)):
    cosine_list=list(similarity_mat[:,program_id])
    cosine_max_number, cosine_max_index = find_max_index(cosine_list, 3)
    recom_job = [school_name[program_id],program_name[program_id]]
    for item in cosine_max_index:
        recom_job.append(job_title[item])
    rmd_job_result.append(recom_job)

import csv
header1 = ['job title', 'recommand1', 'recommand2', 'recommand3']

with open('rmd_program_word2vec_cosine.csv', 'w', newline='') as f1:
    writer = csv.writer(f1)
    writer.writerow(header1)
    writer.writerows(rmd_program_result)

header2 = ['school name','program_name', 'recommand1', 'recommand2', 'recommand3']

with open('rmd_job_word2vec_cosine.csv', 'w', newline='') as f2:
    writer = csv.writer(f2)
    writer.writerow(header2)
    writer.writerows(rmd_job_result)

