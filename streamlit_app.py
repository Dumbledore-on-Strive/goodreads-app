import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()

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
    st.title('Dumbledore on Strive!')  # site title h1
    st.text("Run `/streamlit_app.py` and you'll see the magic :sparkles:")
    st.text(
        "If you have any questions, checkout our [documentation](https://dumbledore-on-strive.github.io/) ")
    st.text("In the meantime, enjoy the journey of:")
    st.text(' ')
    st.markdown('* **Webscraping:** get the information (data) from the web')
    st.markdown('* **Dataframe:** add the data a spreadsheet file')
    st.markdown('* **Plot:** create plot and graphs for better visualization')
    st.text(' ')

    goodreads_data = get_data('data/preprocess_data.csv')
    cleaned = goodreads_data.dropna()

    # header
    st.header("Using magic with Machine Learning to get the the Data we need")
    st.text(' ')
    image = Image.open('imgs/dumbledore-on-strive.jpeg')
    st.image(image, caption="'It takes a great deal of bravery to stand up to our enemies, but just as much to stand up to our friends.'")


with dataset:
    st.header('Goodreads - Books That Should Be Made Into Movies')
    st.text('Data Visualization')

    def load_data():
        df = pd.read_csv('data/preprocess_data.csv')

        return df

    # load dataset
    data = load_data()
    st.write(data)

    # -----------------------

    st.header('Mean Normalization')
    st.text('Plot')

    def mean_norm(data_column_name):
        x = data_column_name
        mean_norm_ratings = 1 + ((x - x.mean()) / (x.max() - x.min())) * 9
        return mean_norm_ratings
    a = mean_norm(cleaned["avg_rating"])
    st.bar_chart(a)
    # st.bar_chart(a).to_frame()

    # ------------------------


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
