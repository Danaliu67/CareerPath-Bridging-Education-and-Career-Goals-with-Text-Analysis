
import numpy as np
import gensim.downloader as api
from gensim.models import KeyedVectors
from utils import find_max_index
import pandas as pd
from utils import skill_classifier
from nltk.tokenize import word_tokenize

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
X = []
labels = []
for test_id in range(0,len(job_data)):
    print(test_id)
    # print(job_data[test_id])
    row = job_data[test_id][1:]
    newlist = [x for x in row if pd.isnull(x) == False]
    newlist_soft, newlist_hard = skill_classifier(newlist, soft_skills)

    cosine_list = []
    euclidean_list = []
    for program in program_data:
        program = program[1:]
      
        program = [x for x in program if pd.isnull(x) == False]

        program_soft, program_hard = skill_classifier(program, soft_skills)
        soft_simi = calc_keyword_similarity(newlist_soft, program_soft, model)
        hard_simi = calc_keyword_similarity(newlist_hard, program_hard, model)
        cosine_list.append(0.5*soft_simi + 0.5*hard_simi)
        

        # cosine_list.append(calc_keyword_similarity(newlist, program, model))
    
    cosine_max_number, cosine_max_index = find_max_index(cosine_list, 3)


    recom = [job_title[test_id]]
    print(job_title[test_id])
    print("matched programs:",end=' ')
    for item in cosine_max_index:
        print("university name:",school_name[item],end=' ')
        print("program name:", program_name[item])
        recom.append(school_name[item] + '/' + program_name[item])

    best_score += 2
    cosine_score += 3 - len(set(cluster_data[cosine_max_index]))
   

    rmd_result.append(recom)

   
print(cosine_score)

print(0, best_score)
print(cosine_score/best_score)

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


