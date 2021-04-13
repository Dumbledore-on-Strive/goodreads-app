from collections import namedtuple
import altair as alt
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import streamlit as st
from PIL import Image

# """
# # Dumbledore on Strive!

# Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

# If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
# forums](https://discuss.streamlit.io).

# In the meantime, below is an example of what you can do with just a few lines of code:
# """

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
comparison = st.beta_container()

with header:
    st.title('Dumbledore welcomes you to our Project')
    st.text('Webscraping, Dataframe, Plot, ')

    st.write("Using magic with Machine Learning to Find the the Data we need")
    image = Image.open('dumbledore-on-strive.jpeg')
    st.image(image, caption='It takes a great deal of bravery to stand up to our enemies, but just as much to stand up to our friends.')


with dataset:
    st.header('Goodreads - Books That Should Be Made Into Movies')
    st.text('Data Visualization')
    # st.text('Spreadsheet Example')
    df_100_rows = pd.read_csv('data/100_rows.csv')
    # st.write(df_100_rows.head())

    st.text('Average Ratings')
    avg_ratings = pd.DataFrame(df_100_rows['Average_Ratings'].value_counts()).head(50)
    st.bar_chart(avg_ratings)

    st.text('Average Ratings')
    avg_ratings = pd.DataFrame(df_100_rows['Average_Ratings'].value_counts()).head(50)
    st.bar_chart(avg_ratings)


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



# with comparison:
#     st.header('Comparison')
#     st.text('The probability of a book with mare than 4 average rating on Goodreads hit average rating higher than 7 on IMDB')
#

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
