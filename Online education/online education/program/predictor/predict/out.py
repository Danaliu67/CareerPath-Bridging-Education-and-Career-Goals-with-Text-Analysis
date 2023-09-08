import streamlit as st
import numpy as np
import pandas as pd
from key_extractor import key_extractor
from predict_job import Job_prediction
from predict_program import Program_prediction
import re
import time
import string

def Title():
	st.title("Online Education(Art)")
	st.header('group5.1')
	st.subheader('2023/4/21')

def Preprocessing(pre_df,pre_df2):
	st.write("Here's our first attempt at using data to do the preprocssing on job:")
	st.write(pre_df)

	st.write("Here's our first attempt at using data to do the preprocssing on program:")
	st.write(pre_df2)

def Choose():
	add_selectbox = st.sidebar.selectbox(
		"Which recommandation would you want?",
		("Job", "Program")
	)
	st.sidebar.subheader("About")
	st.sidebar.info("This is a smart recommender system connecting courses and jobs,\
	                recommending suitable jobs or courses to people based on the description provided,\
					You only need to select the purpose of recommendation, input the text information, \
					click get keywords, the top ten keywords can be displayed, and click get recommendation, you can get corresponding recommendations.")
	return add_selectbox

def Keywords1(key_df1,pre_df):
	option = st.selectbox(
		'Which job you want to choose for the keywords?',
		pre_df['job title'])

	st.write('You selected:', option)
	pre_data = pre_df[pre_df['job title'] == option]
	original = pre_data.iloc[:,1]
	text = original.iloc[0]
	idx = original.index[0]
	st.text('The selected words are:',text)

def Keywords2(opt='Job'):
	st.subheader('Find the keywords from a descriotion')
	if opt == 'Job':
		description = st.text_area('Please input a program description',
									"A course exploring the creation and design of handmade books, including bookbinding, letterpress printing, and artist's books as a form of artistic expression.")
	else:
		description = st.text_area('Please input a job description',
									"Diverse role with close cooperation with the Managing Director and team.Fantastic team culture.Have oversight over a diverse range of accounts.")

	key_etc = key_extractor()
	keys = key_etc.get_key(description)

	if st.button('Get keywords'):
		st.write('The keywords of this description is:', keys)
		description = description.translate(str.maketrans('', '', string.punctuation + string.digits))
		for word in keys:
			match = re.search(word, description.lower())
			#st.write(match)
			start = match.span()[0]
			end = match.span()[1]
			description = description[:start] + "**" + word + "**" + description[end:]

		#st.write('The keywords of this description is:',keys)
		st.info(description)
		#Predict(opt)

	return keys

def Predict(opt):
	st.subheader('To do the recommendation')
	if st.button('Get recommendation'):
		if opt == 'Program':
		    with st.spinner('Wait for it...'):
			    predictor = Job_prediction()
			    result = predictor.predict(keys)
			    time.sleep(3)
		    #st.success('Done!')
		    out_predict = 'The most recommended job is '+str(result[0])+',\n and '+str(result[1])+' and '+str(result[2])+' are also recommended'
		    st.success(out_predict)
		else:
			with st.spinner('Wait for it...'):
				predictor = Program_prediction()
				result = predictor.predict(keys)
				time.sleep(3)
		    #st.write('The recommand programs are:',predictor.predict(keys))
			out_predict = 'The most recommended program is ' + str(result[0]) + ',\n and ' + str(result[1]) + ' and ' + str(result[2]) +' are also recommended'
			st.success(out_predict)


pre_df = pd.read_csv(r'C:\Users\cream\Desktop\datamining\job data_pre(1).csv')#job
pre_df2 = pd.read_csv(r'C:\Users\cream\Desktop\datamining\program data_pre(1).csv',)
key_df = pd.read_csv(r'C:\Users\cream\Desktop\datamining\keywords job1.csv')

Title()
#Preprocessing(pre_df,pre_df2)
opt = Choose()
#Keywords1(key_df,pre_df)
keys = Keywords2(opt)
Predict(opt)

#streamlit run C:/Users/cream/Desktop/datamining/out.py
#"Diverse role with close cooperation with the Managing Director and team.Fantastic team culture.Have oversight over a diverse range of accounts."
#"We are looking for a creative Graphic Designer & Video Editor! We are searching for a talented and ambitious Graphic Designer & Video Editor who is able to create content that takes our beauty brands to the next level."