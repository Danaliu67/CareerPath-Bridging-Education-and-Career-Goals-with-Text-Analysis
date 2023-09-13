import numpy as np
import fasttext
import pandas as pd

from ..utils import find_max_index


def download_model(model_path):
    import urllib.request
    import os

    if not os.path.exists(model_path):
        print(f"Downloading model to {model_path}...")
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
        urllib.request.urlretrieve(url, model_path + ".gz")
        import gzip
        with gzip.open(model_path + ".gz", "rb") as f_in:
            with open(model_path, "wb") as f_out:
                f_out.write(f_in.read())
        os.remove(model_path + ".gz")
        print("Model downloaded successfully.")

def calc_fasttext_similarity(word_list1, word_list2, model):
    
    embeddings1 = np.zeros((len(word_list1), model.get_dimension()))
    embeddings2 = np.zeros((len(word_list2), model.get_dimension()))

    for i, word in enumerate(word_list1):
        embeddings1[i] = model.get_word_vector(word)
    for i, word in enumerate(word_list2):
        embeddings2[i] = model.get_word_vector(word)

    similarities = np.dot(embeddings1, embeddings2.T) / (
        np.linalg.norm(embeddings1, axis=1)[:, None] * np.linalg.norm(embeddings2, axis=1)[None, :]
    )
    avg_similarity = np.max(np.mean(similarities, axis=0))
    return avg_similarity


model_path = "../checkpoints/cc.en.300.bin"
download_model(model_path)
model = fasttext.load_model(model_path)

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

# df_job = pd.read_csv("../../../eval/eval/clustered_data_job.csv") 
# cluster_data_job = df_job["cluster"].values

# df_program = pd.read_csv("../../../eval/eval/clustered_data_program.csv") 
# cluster_data_program = df_program["cluster"].values
df3 = pd.read_csv("../../../eval/eval/clustered_data.csv") 
cluster_data = df3["cluster"].values


rmd_result = []
cosine_score = 0
jaccard_score = 0
euclidean_score = 0
best_score = 0
worst_score = 0

for test_id in range(0,len(job_data)):
#     print(test_id)
    row = job_data[test_id][1:]
    newlist = [x for x in row if pd.isnull(x) == False]
  

    cosine_list = []
    jaccard_list = []
    euclidean_list = []
    for program in program_data:
        program = program[1:]
        program = [x for x in program if pd.isnull(x) == False]
        cosine_list.append(calc_fasttext_similarity(newlist, program, model))

    cosine_max_number, cosine_max_index = find_max_index(cosine_list, 3)
   
    recom = [job_title[test_id]]
    for item in cosine_max_index:
        recom.append(school_name[item] + '/' + program_name[item])
    cosine_score+=3-len(set(cluster_data[cosine_max_index]))
    worst_score += 3


    rmd_result.append(recom)

print(cosine_score)
print(worst_score)

import csv
header = ['job title', 'recommand1', 'recommand2', 'recommand3']

with open('rmd_program_fasttext_cosine.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rmd_result)
