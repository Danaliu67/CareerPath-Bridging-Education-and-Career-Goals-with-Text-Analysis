import re
import time
import string
import streamlit as st

from OnlineEdu.text import KeyExtractor
from OnlineEdu.model import JobPrediction, ProgramPrediction


def Title():
	st.title("Online Education (Art)")
	st.header('2023/4/21')


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


def Keywords(opt='Job'):
	st.subheader('Find the keywords from a descriotion')
	if opt == 'Job':
		description = st.text_area('Please input a program description',
									"A course exploring the creation and design of handmade books, including bookbinding, letterpress printing, and artist's books as a form of artistic expression.")
	else:
		description = st.text_area('Please input a job description',
									"Diverse role with close cooperation with the Managing Director and team.Fantastic team culture.Have oversight over a diverse range of accounts.")

	key_etc = KeyExtractor()
	keys = key_etc.get_key(description)

	if st.button('Get keywords'):
		st.write('The keywords of this description is:', keys)
		description = description.translate(str.maketrans('', '', string.punctuation + string.digits))
		for word in keys:
			match = re.search(word, description.lower())
			start = match.span()[0]
			end = match.span()[1]
			description = description[:start] + "**" + word + "**" + description[end:]
		st.info(description)
	return keys

def Predict(opt, keys):
	st.subheader('To do the recommendation')
	if st.button('Get recommendation'):
		if opt == 'Job':
			with st.spinner('Wait for it...'):
				predictor = JobPrediction()
				result = predictor.predict(keys)
				time.sleep(3)
			out_predict = 'The most recommended job is '+str(result[0])+',\n and '+str(result[1])+' and '+str(result[2])+' are also recommended'
			st.success(out_predict)
		else:
			with st.spinner('Wait for it...'):
				predictor = ProgramPrediction()
				result = predictor.predict(keys)
				time.sleep(3)
			out_predict = 'The most recommended program is ' + str(result[0]) + ',\n and ' + str(result[1]) + ' and ' + str(result[2]) +' are also recommended'
			st.success(out_predict)


# if __name__ == "main":
Title()
opt = Choose()
keys = Keywords(opt)
Predict(opt, keys)
