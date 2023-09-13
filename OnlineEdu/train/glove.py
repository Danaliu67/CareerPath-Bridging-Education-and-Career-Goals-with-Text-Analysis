import gensim.downloader as api
import pandas as pd
import csv

from ..utils import find_max_index

 
def load_glove_model():
    glove_model = api.load("glove-wiki-gigaword-100")
    return glove_model


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

for test_id in range(0,len(job_data)):
    row = job_data[test_id][1:]
    newlist = [x for x in row if pd.isnull(x) == False]

    cosine_list = []
    jaccard_list = []
    euclidean_list = []
    for program in program_data:
        program = program[1:]
        program = [x for x in program if pd.isnull(x) == False]
        cosine_list.append(calc_keyword_similarity(newlist, program, model))
    cosine_max_number, cosine_max_index = find_max_index(cosine_list, 3)
   
    recom = [job_title[test_id]]
    for item in cosine_max_index:
        recom.append(school_name[item] + '/' + program_name[item])
    best_score += 1
    cosine_score+=3-len(set(cluster_data[cosine_max_index]))
    worst_score += 2
    rmd_result.append(recom)

print(cosine_score)
print(worst_score)
print(cosine_score/worst_score)

header = ['job title', 'recommand1', 'recommand2', 'recommand3']

with open('rmd_program_glove_cosine.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rmd_result)
