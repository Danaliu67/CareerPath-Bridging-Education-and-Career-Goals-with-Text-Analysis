import pandas as pd
import numpy as np
import copy


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

def cal_similarity(bowA, bowB, type_ = 'cosine',v_type='BOW'):
    if v_type=='BOW':
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
        return cosine(count_vector[0],count_vector[1])
    elif(type_ == 'euclidean'):
        return Distance(count_vector[0],count_vector[1])
    else:
        print("lack para")
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
#     print(test_id)
    row = job_data[test_id][1:]
    newlist = [x for x in row if pd.isnull(x) == False]
  

    cosine_list = []
    jaccard_list = []
    euclidean_list = []
    for program in program_data:
        program = program[1:]
        program = [x for x in program if pd.isnull(x) == False]
        cosine_list.append(cal_similarity(newlist, program))

    cosine_max_number, cosine_max_index = find_max_index(cosine_list, 3)
   
    recom = [job_title[test_id]]
    for item in cosine_max_index:
        recom.append(school_name[item] + '/' + program_name[item])
    cosine_score += 3-len(set(cluster_data[cosine_max_index]))
    worst_score += 2



    rmd_result.append(recom)

print(cosine_score)
print(worst_score)
print(cosine_score/worst_score)
import csv
header = ['job title', 'recommand1', 'recommand2', 'recommand3']

with open('rmd_program_bow_cosine.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rmd_result) 
