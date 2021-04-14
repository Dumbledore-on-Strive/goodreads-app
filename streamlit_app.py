from collections import namedtuple
import altair as alt
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler
from bs4 import BeautifulSoup
import streamlit as st
from PIL import Image

"""
# Dumbledore on Strive!

Run `/streamlit_app.py` and you'll see the magic :sparkles:

If you have any questions, checkout our [documentation](https://dumbledore-on-strive.github.io/) 

In the meantime, enjoy the journey of:
"""

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
comparison = st.beta_container()

st.markdown(
    """
    <style>
     .main {
     # background-color: #0d0e12;

     }
    </style>
    """,
    unsafe_allow_html=True
)

# @st.cache
def get_data(file):
    goodreads_data = pd.read_csv(file)

    return goodreads_data

with header:
    # st.title('Dumbledore welcomes you to our Project')
    st.text('Webscraping, Dataframe, Plot')

    goodreads_data = get_data('data/df-copy-1.csv')

    st.write("Using magic with Machine Learning to get the the Data we need")
    image = Image.open('dumbledore-on-strive.jpeg')
    st.image(image, caption="'It takes a great deal of bravery to stand up to our enemies, but just as much to stand up to our friends.'")


with dataset:
    st.header('Goodreads - Books That Should Be Made Into Movies')
    st.text('Data Visualization')
    st.text('Spreadsheet Example')
    goodreads_data = get_data('data/df-copy-1.csv')
    # st.write(taxi_data.head())

    #------------------------

    st.header('2D scatterplot with pages and number of ratings')
    cleaned.plot.scatter('num_pages', 'num_rating', color='k', edgecolor='r')
    plt.xlabel('num_pages')
    plt.ylabel('num_rating')
    plt.title('2D Scatter plot')
    # plt.legend()
    plt.grid()
    plt.show()


    # ------------------------

    st.header('Matplotlib')
    arr = np.random.normal(1, 1, size=100)
    fig, ax = plt.subplots()#0d0e12
    ax.hist(arr, bins=20)
    # fig.patch.set_facecolor('#0d0e12')
    # ax.set_facecolor('#0d0e12')
    st.pyplot(fig)

    # ------------------------

    st.header('Min-Max Normalization')
    cleaned = goodreads_data.dropna()
    # Step 1: find min(avg_rating)
    minValue = cleaned[['avg_rating']].min()
    st.text(minValue)
    # st.text('minimum rating: ', minValue)

    # Step2: find max(avg_rating)
    maxValue = cleaned[['avg_rating']].max()
    st.text(maxValue)
    # st.text('maximum rating', maxValue)

    # ------------------------

    st.header('Mean Normalization')

    def mean_norm(data_column_name):
        x = data_column_name
        mean_norm_ratings = 1 + ((x - x.mean()) / (x.max() - x.min())) * 9
        return mean_norm_ratings
    a = mean_norm(cleaned["avg_rating"])
    st.bar_chart(a)
    # st.bar_chart(a).to_frame()

    # ------------------------

    # average (avg_rating)
    df_mean = cleaned[["avg_rating"]].mean()
    st.text(df_mean)

    # ------------------------

    st.header('Average Ratings')
    avg_ratings = pd.DataFrame(goodreads_data['avg_rating'].value_counts()).head(50)
    st.bar_chart(avg_ratings)

    st.header('Original Publish Year')
    original_publish_year = pd.DataFrame(goodreads_data['original_publish_year'].value_counts()).head(50)
    st.line_chart(original_publish_year)

    st.header('Number of Pages')
    num_pages = pd.DataFrame(goodreads_data['num_pages'].value_counts()).head(50)
    st.line_chart(num_pages)


with features:
    st.header('Features')
    st.text(
        '- Create a 2D scatterplot with pages on the x-axis and num_ratings on the y-axis.')
    st.text('- compute numerically the correlation')
    st.text('- Visualise the avg_rating distribution')
    st.text('- Visualise the minmax_norm_rating distribution.')
    st.text('- Visualise the mean_norm_rating distribution.')
    st.text('- Create one graph that represents in the same figure both minmax_norm_rating and mean_norm_ratingdistributions.')
    st.text('- Scipy-Stats to represent the best fit in terms of a distribution')
    st.text('- Visualize the awards distribution in a boxplot and aggregated bars')
    st.text('- Group the books by original_publish_year and get the mean of the minmax_norm_ratings of the groups')
    st.text('- Make a scatterplot to represent  minmax_norm_ratings in function of the number of awards won by the book')


with comparison:
    st.header('Comparison')
    st.text('The probability of a book with mare than 4 average rating on Goodreads hit average rating higher than 7 on IMDB')


# with st.echo(code_location='below'):
#     total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
#     num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

#     Point = namedtuple('Point', 'x y')
#     data = []

#     points_per_turn = total_points / num_turns

#     for curr_point_num in range(total_points):
#         curr_turn, i = divmod(curr_point_num, points_per_turn)
#         angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
#         radius = curr_point_num / total_points
#         x = radius * math.cos(angle)
#         y = radius * math.sin(angle)
#         data.append(Point(x, y))

#     st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
#                     .mark_circle(color='#0068c9', opacity=0.5)
#                     .encode(x='x:Q', y='y:Q'))
