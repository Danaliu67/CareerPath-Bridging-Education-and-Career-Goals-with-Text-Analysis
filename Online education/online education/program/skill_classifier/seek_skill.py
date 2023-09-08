import requests
from bs4 import BeautifulSoup
import csv
import re
import time
import chardet

# 构造请求头，避免被网站识别为爬虫
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

# 构造CSV文件头
csv_header = ['skill']

# 遍历每个页面
# results = []
# url = f'https://www.ctgoodjobs.hk/ctjob/listing/joblist.asp?job_area=013_jc,003_jc,043_jc'
all_skills = []
for page in range(0,1):
    print("--------------scrabing page:", page, '------------------')

    url = f'https://www.onetonline.org/skills/soft/result?s=2.B.1.b&s=2.B.1.e&s=2.B.1.d&s=2.B.1.c&s=2.B.1.f&s=2.B.1.a&s=2.A.2.b&s=2.A.1.b&s=2.B.2.i&s=2.A.2.a&s=2.B.4.e&s=2.A.2.c&s=2.A.2.d&s=2.B.5.a{page}'
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    job_links = []
    occupations = soup.find_all('td', {'data-title': 'Occupation'})

    for occupation in occupations:
        href = occupation.find('a')['href']
        job_links.append(href)

    
    for link in job_links:
        print(link)
        # job_url = link.get('href')
        job_response = requests.get(link, headers=headers)
        job_soup = BeautifulSoup(job_response.text, 'html.parser')

        # 找到<h2>标签
        tag = job_soup.find('h2', class_="r-group-header fs-4", text='Worker Requirements')
        # print(tag)
        # work_requirements = h6_tag.find_next_all()

        skills = tag.find_all_next('div', class_='order-2 flex-grow-1')
        # skill = work_requirements.find('div', class_='order-2 flex-grow-1')
        # print(skill)
        for skill in skills:
            all_skills.append(skill.b.get_text())
            print(all_skills[-1])

        time.sleep(3)

all_skills = list(set(all_skills))
print(len(all_skills))
with open('soft_skills_3.txt', 'w') as f:
    for element in all_skills:
        f.write(element + '\n')


print('over!')
