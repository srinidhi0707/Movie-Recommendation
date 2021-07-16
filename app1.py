import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

st.set_page_config(page_title ="Movie Recommendation",page_icon="🎥")
###### helper functions. Use them when needed #######
def get_title_from_index(index):
	return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
	return df[df.title == title]["index"].values[0]
##################################################

##Step 1: Read CSV File
df = pd.read_csv("movie_dataset.csv")
#print df.columns
##Step 2: Select Features

features = ['keywords','cast','genres','director']
##Step 3: Create a column in DF which combines all selected features
for feature in features:
	df[feature] = df[feature].fillna('')

def combine_features(row):
	try:
		return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
	except:
		print("Error:", row)	

df["combined_features"] = df.apply(combine_features,axis=1)

#print "Combined Features:", df["combined_features"].head()
def recommend_movies(movie_user_likes):

##Step 4: Create count matrix from this new combined column
       cv = CountVectorizer()

       count_matrix = cv.fit_transform(df["combined_features"])

##Step 5: Compute the Cosine Similarity based on the count_matrix
       cosine_sim = cosine_similarity(count_matrix) 

## Step 6: Get index of this movie from its title
       movie_index = get_index_from_title(movie_user_likes)

       similar_movies =  list(enumerate(cosine_sim[movie_index]))
 
## Step 7: Get a list of similar movies in descending order of similarity score
       sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)
       Recomended_movies=[]

## Step 8: Print titles of first 50 movies
       i=0
       for element in sorted_similar_movies:
                a=get_title_from_index(element[0]) 
                print(a)
                Recomended_movies.append(a)
                i=i+1
                if i>20:
                    break
       return Recomended_movies


def main():
    html_temp = """
        <div style="background-color:Grey;padding:10px">
        <h1 style="color:white;text-align:center;">Movie Recommendation </h1>
        </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    movie_user_likes  = st.text_input("Enter the name of the movie:")
    st.success(f'The movies similar to {movie_user_likes} are')

    print(movie_user_likes)
    if st.button("Recommend"):
        result = recommend_movies(movie_user_likes)
        
       #  st.table(result)
        st.table(result)
        
        

if __name__=='__main__':
    main()