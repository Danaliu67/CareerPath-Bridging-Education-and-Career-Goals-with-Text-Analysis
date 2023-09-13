
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors
from utils import find_max_index
import pandas as pd

# 下载GloVe模型并加载
def load_glove_model():
    glove_model = api.load("glove-wiki-gigaword-100")
    return glove_model

# 计算两个关键词列表的相似度
def calc_keyword_similarity(keyword_list_1, keyword_list_2, model):
    similarity = 0
    count = 0
    for keyword_1 in keyword_list_1:
        for keyword_2 in keyword_list_2:
            try:
                similarity += model.similarity(keyword_1, keyword_2)
                count += 1
            except KeyError:
                continue
    if count > 0:
        similarity /= count
    return similarity



model = load_glove_model()

# read job
file_path = 'keywords job(ver2).xls'   
raw_data = pd.read_excel(file_path, header=0)  
job_data = raw_data.values

# read program
program_file = 'keywords_program(2).xlsx'   
program_raw_data = pd.read_excel(program_file, header=0)  
program_data = program_raw_data.values
program_row = program_data[0][1:]
# print(len(program_data))

# to get job and program name
# read job
df1 = pd.read_csv("job data_pre(ver2).csv")
job_title = df1['job title']

# read program
df2 = pd.read_csv("program data_pre(ver2).csv")
school_name = df2['Schoole']
program_name = df2['program']

df3 = pd.read_csv("../../../eval/eval/clustered_data.csv") 
cluster_data = df3["cluster"].values
# df_job = pd.read_csv("../../../eval/eval/clustered_data_job.csv") 
# cluster_data_job = df_job["cluster"].values

# df_program = pd.read_csv("../../../eval/eval/clustered_data_program.csv") 
# cluster_data_program = df_program["cluster"].values

rmd_result = []
cosine_score = 0
jaccard_score = 0
euclidean_score = 0
best_score = 0
worst_score = 0

for test_id in range(0,len(job_data)):
#     print(test_id)
    # print(job_data[test_id])
    row = job_data[test_id][1:]
    # newlist = [x if pd.isnull(x) == False else '<pad>' for x in row]
    newlist = [x for x in row if pd.isnull(x) == False]
    # print(newlist)

    cosine_list = []
    jaccard_list = []
    euclidean_list = []
    for program in program_data:
        # print(program)
        program = program[1:]
        # print("here\n",newlist)
        # print(program)
        # program = [x if pd.isnull(x) == False else '<pad>' for x in program]
        program = [x for x in program if pd.isnull(x) == False]

        cosine_list.append(calc_keyword_similarity(newlist, program, model))
        # cosine_list.append(cal_similarity(newlist, program[1:]))
        # jaccard_list.append(Jaccard(newlist, program[1:]))
        # euclidean_list.append(cal_similarity(newlist, program[1:], type_='euclidean'))
        # print("cosine:",cal_similarity(newlist, program[1:]),end='\t')
        # print('Jaccard:',Jaccard(newlist, program[1:]), end='\t')
        # print('Euclidean distance:',cal_similarity(newlist, program[1:], type_='euclidean'))

    # print(cosine_list)
    cosine_max_number, cosine_max_index = find_max_index(cosine_list, 3)
#     cosine_max_index=cosine_list.index(max(cosine_list))
    # jaccard_max_number, jaccard_max_index = find_max_index(jaccard_list, 3)
    # euclidean_max_number, euclidean_max_index = find_max_index(euclidean_list, 3)
    # print(cosine_max_number)
    # print(cosine_max_index)

   
    recom = [job_title[test_id]]
#     print(job_title[test_id])
#     print("matched programs:",end=' ')
    for item in cosine_max_index:
#         print("university name:",school_name[item],end=' ')
#         print("program name:", program_name[item])
#     item=cosine_max_index
        recom.append(school_name[item] + '/' + program_name[item])
    best_score += 1
#     cosine_score += 3 - len(set(cosine_max_index))
    # jaccard_score += 3 - len(set(jaccard_max_index))
    # euclidean_score += 3 - len(set(euclidean_max_index))
#     print(cluster_data_job[test_id])
#     print(cluster_data_program[cosine_max_index])
#     if cluster_data_job[test_id]!=cluster_data_program[cosine_max_index]:
#         cosine_score += 1
    cosine_score+=3-len(set(cluster_data[cosine_max_index]))
    worst_score += 2


    rmd_result.append(recom)

# print(rmd_result)
print(cosine_score)
# print(jaccard_score)
# print(euclidean_score)
print(worst_score)
print(cosine_score/worst_score)

import csv
header = ['job title', 'recommand1', 'recommand2', 'recommand3']

with open('rmd_program_glove_cosine.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rmd_result)






# keyword_list_1 = ['apple', 'fruit', 'juice']
# keyword_list_2 = ['orange', 'drink']
# similarity = calc_similarity(keyword_list_1, keyword_list_2, model)
# print(similarity)


