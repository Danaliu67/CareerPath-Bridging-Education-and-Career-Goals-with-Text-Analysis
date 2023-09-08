import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words('english')


df = pd.read_csv('archive-2/skills_index_final.csv')
# 将短语转换成单词列表
all_words = []
for skill in df['Skill']:
    words = word_tokenize(skill.lower())
    all_words.extend(words)


#---------------------
with open('softskill.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]

all_words2 = []
for line in lines:
    words = word_tokenize(line.lower())
    # 将单词添加到all_words列表中
    all_words2.extend(words)


#-----------------
with open('soft_skills_3.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]

all_words3 = []
for line in lines:
    words = word_tokenize(line.lower())
    all_words3.extend(words)


soft_skill_words = list(set(all_words))
print(len(soft_skill_words))

soft_skill_words2 = list(set(all_words2))
print(len(soft_skill_words2))

soft_skill_words3 = list(set(all_words3))
print(len(soft_skill_words3))


soft_skill_words.extend(soft_skill_words2)
soft_skill_words.extend(soft_skill_words3)

soft_skill_words = [word for word in soft_skill_words if word not in stop_words]


print(len(soft_skill_words))


with open('all_soft_skills.txt', 'w') as f:
    for element in soft_skill_words:
        f.write(element + '\n')


# read job
file_path = '../data_key/keywords job(ver2).xls'   
raw_data = pd.read_excel(file_path, header=0)  
job_data = raw_data.values
job_words = []
for row in job_data:
    job_words.extend(row[1:])

# read program
program_file = '../data_key/keywords_program(2).xlsx'   
program_raw_data = pd.read_excel(program_file, header=0)  
program_data = program_raw_data.values
program_words= []
for row in program_data:
    program_words.extend(row[1:])



# 去除重复单词
unique_job_words = list(set(job_words))
print("job words:",len(unique_job_words))
unique_pro_words = list(set(program_words))
print("program words:",len(unique_pro_words))

labeled_soft_skill = []
for word in unique_job_words:
    if word in soft_skill_words:
        labeled_soft_skill.append(word)
# print(labeled_soft_skill)
print("soft skill of job:",len(labeled_soft_skill))

labeled_soft_skill_p = []
for word in unique_pro_words:
    if word in soft_skill_words:
        labeled_soft_skill_p.append(word)
# print(labeled_soft_skill)
print("soft skill of program:",len(labeled_soft_skill_p))