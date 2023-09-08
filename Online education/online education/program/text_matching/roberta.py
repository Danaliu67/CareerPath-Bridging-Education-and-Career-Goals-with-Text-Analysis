from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
import torch
import numpy as np
import pandas as pd
import copy


config=RobertaConfig.from_pretrained('roberta-base')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base',config=config)



def calc_keyword_similarity(keyword_vectors_1, keyword_vectors_2):
    # 计算两个关键词列表的余弦相似度
    sim = np.dot(keyword_vectors_1, keyword_vectors_2.T) / \
          (np.linalg.norm(keyword_vectors_1) * np.linalg.norm(keyword_vectors_2))
#     print(sim)
    return sim


def find_max_index(list_, max_num=3):
    t = copy.deepcopy(list_)
    max_number = []
    max_index = []
    for _ in range(max_num):
        number = max(t)
        index = t.index(number)
        t[index] = 0
        max_number.append(number)
        max_index.append(index)
    t = []
    # print(max_number)
    # print(max_index)
    if(0.0 in max_number):
        discard = []
        for i in range(len(max_number)):
            if max_number[i] == 0.0:
                discard.append(max_index[i])

        max_number.remove(0.0)
        for item in discard:
            max_index.remove(item)

    return (max_number, max_index)


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

rmd_result = []
cosine_score = 0
jaccard_score = 0
euclidean_score = 0
best_score = 0
worst_score = 0

job_vectors=[]
program_vectors=[]
for i in range(len((job_data))):
    row = job_data[i][1:]
    newlist = [x for x in row if pd.isnull(x) == False]
    input_1=tokenizer(' '.join(newlist),return_tensors="pt")
    output_1=model(**input_1)
    keyword_vectors_1 = torch.mean(output_1.last_hidden_state, dim=1).detach().numpy()
    job_vectors.append(keyword_vectors_1)

for j in range(len(program_data)):
    row = program_data[j][1:]
    newlist = [x for x in row if pd.isnull(x) == False]
    input_2=tokenizer(' '.join(newlist),return_tensors="pt")
    output_2=model(**input_2)
    keyword_vectors_2 = torch.mean(output_2.last_hidden_state, dim=1).detach().numpy()
    program_vectors.append(keyword_vectors_2)
    
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
        # print(program)
        program = program_data[program_id][1:]
        program = [x for x in program if pd.isnull(x) == False]
        similarity=calc_keyword_similarity(job_vectors[test_id], program_vectors[program_id])
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
# for test_id in range(0,len(job_data)):
#     cosine_list = []
#     jaccard_list = []
#     euclidean_list = []
#     for program_id in range(len(program_data)):
#         cosine_list.append(calc_bert_keyword_similarity(job_vectors[test_id], program_vectors[program_id]))

#     cosine_max_number, cosine_max_index = find_max_index(cosine_list, 3)
#     recom = [job_title[test_id]]
#     for item in cosine_max_index:
#         recom.append(school_name[item] + '/' + program_name[item])
#     cosine_score += 3-len(set(cluster_data[cosine_max_index]))
#     worst_score += 2
#     rmd_result.append(recom)
# print(cosine_score)
# print(worst_score)
# print(cosine_score/worst_score)
import csv
header1 = ['job title', 'recommand1', 'recommand2', 'recommand3']

with open('rmd_program_roberta_cosine.csv', 'w', newline='') as f1:
    writer = csv.writer(f1)
    writer.writerow(header1)
    writer.writerows(rmd_program_result)

header2 = ['school name','program_name', 'recommand1', 'recommand2', 'recommand3']

with open('rmd_job_roberta_cosine.csv', 'w', newline='') as f2:
    writer = csv.writer(f2)
    writer.writerow(header2)
    writer.writerows(rmd_job_result)