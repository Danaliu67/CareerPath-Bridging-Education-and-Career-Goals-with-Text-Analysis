from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
import torch
import numpy as np
import pandas as pd
import copy
from utils import skill_classifier
from utils import find_max_index
from nltk.tokenize import word_tokenize

config=RobertaConfig.from_pretrained('roberta-base')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base',config=config)



def calc_bert_keyword_similarity(keyword_vectors_1, keyword_vectors_2):
    # 计算两个关键词列表的余弦相似度
    sim = np.dot(keyword_vectors_1, keyword_vectors_2.T) / \
          (np.linalg.norm(keyword_vectors_1) * np.linalg.norm(keyword_vectors_2))
#     print(sim)
    return sim


## read job
file_path = '../../data_key/keywords job(ver2).xls'   
raw_data = pd.read_excel(file_path, header=0)  
job_data = raw_data.values

# read program
program_file = '../../data_key/keywords_program(2).xlsx'   
program_raw_data = pd.read_excel(program_file, header=0)  
program_data = program_raw_data.values
program_row = program_data[0][1:]
print(len(program_data))

with open('../../skill_classifier/all_soft_skills.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]
soft_skills = []
for line in lines:
    words = word_tokenize(line.lower())
    soft_skills.extend(words)


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

job_vectors=[]
job_vectors_soft = []
job_vectors_hard = []

program_vectors=[]
program_vectors_soft = []
program_vectors_hard = []

for i in range(len((job_data))):
    row = job_data[i][1:]
    newlist = [x for x in row if pd.isnull(x) == False]
    newlist_soft, newlist_hard = skill_classifier(newlist, soft_skills)

    input_1_soft=tokenizer(' '.join(newlist_soft),return_tensors="pt")
    output_1_soft=model(**input_1_soft)
    keyword_vectors_1_soft = torch.mean(output_1_soft.last_hidden_state, dim=1).detach().numpy()
    job_vectors_soft.append(keyword_vectors_1_soft)

    input_1_hard=tokenizer(' '.join(newlist_hard),return_tensors="pt")
    output_1_hard=model(**input_1_hard)
    keyword_vectors_1_hard = torch.mean(output_1_hard.last_hidden_state, dim=1).detach().numpy()
    job_vectors_hard.append(keyword_vectors_1_hard)


    
for j in range(len(program_data)):
    row = program_data[j][1:]
    newlist = [x for x in row if pd.isnull(x) == False]

    newlist_soft, newlist_hard = skill_classifier(newlist, soft_skills)

    input_2_soft=tokenizer(' '.join(newlist_soft),return_tensors="pt")
    output_2_soft=model(**input_2_soft)
    keyword_vectors_2_soft = torch.mean(output_2_soft.last_hidden_state, dim=1).detach().numpy()
    program_vectors_soft.append(keyword_vectors_2_soft)

    input_2_hard=tokenizer(' '.join(newlist_hard),return_tensors="pt")
    output_2_hard=model(**input_2_hard)
    keyword_vectors_2_hard = torch.mean(output_2_hard.last_hidden_state, dim=1).detach().numpy()
    program_vectors_hard.append(keyword_vectors_2_hard)


for test_id in range(0,len(job_data)):
    cosine_list = []
    jaccard_list = []
    euclidean_list = []
    for program_id in range(len(program_data)):
        soft_simi = calc_bert_keyword_similarity(job_vectors_soft[test_id], program_vectors_soft[program_id])
        hard_simi = calc_bert_keyword_similarity(job_vectors_hard[test_id], program_vectors_hard[program_id])
        cosine_list.append(0.5*soft_simi + 0.5*hard_simi)

        # cosine_list.append(calc_bert_keyword_similarity(job_vectors[test_id], program_vectors[program_id]))

    cosine_max_number, cosine_max_index = find_max_index(cosine_list, 3)
    recom = [job_title[test_id]]
    for item in cosine_max_index:
        recom.append(school_name[item] + '/' + program_name[item])
    cosine_score += 3-len(set(cluster_data[cosine_max_index]))
    worst_score += 2
    rmd_result.append(recom)
print("COSINE",cosine_score)
print(worst_score)
print(cosine_score/worst_score)

# import csv
# header = ['job title', 'recommand1', 'recommand2', 'recommand3']
# with open('rmd_program_bertc_cosine.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(header)
#     writer.writerows(rmd_result)