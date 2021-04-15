import numpy as np
import pandas as pd
import streamlit as st
from sklearn import datasets
from PIL import Image
import seaborn as sns


header = st.beta_container()
dataset = st.beta_container()
comparison = st.beta_container()

# set style for seaborn
sns.set_style('darkgrid')
# set_style("whitegrid")

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
    st.text(" ")
    st.markdown("Enjoy the journey and you'll see the magic :sparkles:")
    st.text(' ')
    st.markdown('* **Webscraping:** get the information (data) from the web')
    st.markdown('* **Dataframe:** add the data a spreadsheet file')
    st.markdown('* **Plot:** create plot and graphs for better visualization')
    st.text(' ')

    goodreads_data = get_data('data/preprocess_data.csv')
    cleaned = goodreads_data.dropna()

    # header
    st.header("Using magic and Machine Learning to get the Data we need")
    st.text(' ')
    image = Image.open('imgs/dumbledore-on-strive.jpeg')
    st.image(image, caption="'It takes a great deal of bravery to stand up to our enemies, but just as much to stand up to our friends.'")

    # button
    if st.button("Let's do this"):
        st.write('Yeah!!!')
    else:
        st.write(' ')

with dataset:
    st.header('Goodreads - Books That Should Be Made Into Movies')
    st.text('Data Visualization')

    def load_data():
        df = pd.read_csv('data/preprocess_data.csv')

        return df

    # # load dataset
    data = load_data()
    # numeric_columns = data.select_dtypes(
    #     ['float64', 'float32', 'int32'])
    # print(numeric_columns)

    # # sidebar
    # # checkbox widget
    # checkbox = st.sidebar.checkbox('Show data')
    # # checkbox = st.checkbox('Show data')
    # print(checkbox)

    # if checkbox:
    #     st.dataframe(data=data)

    # # create scatter plots
    # st.sidebar.subheader('Plot setup')
    # # add select widget
    # select_columns1 = st.sidebar.selectbox(
    #     label='X axis', options=numeric_columns)
    # select_columns2 = st.sidebar.selectbox(
    #     label='Y axis', options=numeric_columns)
    # sns.relplot(x=select_columns1, y=select_columns2, data=data)
    # st.pyplot()

    # -----------------------

    # create scatter plot on seaborn
    # sns.relplot(x=select_columns1, y=select_columns2, data=data)
    # st.pyplot()

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

    # st.header('Features')
    st.markdown(
        '- Create a 2D scatterplot with pages on the x-axis and num_ratings on the y-axis.')
    st.text(" ")

    # def scatter_2D_plot(df):
    #     size_b = df['award']**2*12
    #     colors = np.random.rand(df.shape[0])
    #     sns.scatterplot(df['num_pages'], df['num_rating'],
    #                     s=size_b, c=colors, alpha=0.5, legend=True)
    #     st.pyplot()
    # st.write(scatter_2D_plot(df))
    # st.scatter_plot(df)

    st.markdown('- compute numerically the correlation')
    st.text(" ")

    # def correlation_coff(df):
    #     x = df['num_pages']
    #     y = df['num_rating']
    #     print(
    #         f"Pearson correlation coefficient :{st.pearsonr(x, y)[0] }\nSpearman correlation coefficient :{ st.spearmanr(x, y)[0] }\nKendall’s  correlation coefficient :{st.kendalltau(x, y)[0] }")
    # st.write(correlation_coff(df))

    st.markdown('- Visualise the avg_rating distribution')
    st.text(" ")

    st.markdown('- Visualise the minmax_norm_rating distribution.')
    st.text(" ")

    st.markdown('- Visualise the mean_norm_rating distribution.')
    st.text(" ")

    st.markdown(
        '- Create one graph that represents in the same figure both minmax_norm_rating and mean_norm_ratingdistributions.')
    st.text(" ")

    st.markdown(
        '- Scipy-Stats to represent the best fit in terms of a distribution')
    st.text(" ")

    st.markdown(
        '- Visualize the awards distribution in a boxplot and aggregated bars')
    st.text(" ")

    st.markdown(
        '- Group the books by original_publish_year and get the mean of the minmax_norm_ratings of the groups')
    st.text(" ")

    st.markdown(
        '- Make a scatterplot to represent  minmax_norm_ratings in function of the number of awards won by the book')
    st.text(" ")


# ------------------------------------------------------------------------

st.markdown(
    "If you have any questions, checkout our [documentation](https://dumbledore-on-strive.github.io/) ")
