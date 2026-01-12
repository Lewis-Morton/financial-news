import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from components import overview, data_distributions, categorical_analysis, market_movement_analysis, correlations, nltk, predictive

#load preprocessed datasets and cleaned word lists from jupyter for use in the dashboard
df = pd.read_csv('data/financial_news_events.csv')
df_clean = pd.read_csv('data/financial_news_events_clean.csv')
heatmap_df = pd.read_csv('data/financial_news_events_heatmap_df.csv')
clean_words = pd.read_csv('data/clean_words.csv')['Word'].tolist()
word_series = pd.Series(clean_words)
df_one_hot_encoded = pd.read_csv('data/df_one_hot_encoded.csv')

#without setting the layout to wide, only 6 tabs are rendered
st.set_page_config(layout='wide')

st.title('Financial News Dashboard', width='stretch')
st.divider()
st.header('Analyse the Effects of Financial News on Markets', width='stretch')
st.text('In the following analysis we look at the effects of financial news headlines on markets')
st.text('The aim of the analysis is to uncover market trends which could serve traders who wish factor in the impact of news events on market performance and trading patterns')
#expander keeps key information above the fold
with st.expander('See more details'):
    st.text('Columns in the dataset include: \n-date\n-headline source\n-market event\n-market index\n-index change percent\n-trading volume\n-sector\n-sentiment\n-impact level\n-related company\n-news url')
    

st.sidebar.header('Financial News Controls')

#date range filter
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime(df_clean['Date']).min())
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime(df_clean['Date']).max())

#sentiment filter
selected_sentiments = st.sidebar.multiselect(
    "Sentiment",
    options=df_clean['Sentiment'].unique(),
    default=df_clean['Sentiment'].unique()
)

#sector filter
selected_sectors = st.sidebar.multiselect(
    "Sector",
    options=df_clean['Sector'].unique(),
    default=df_clean['Sector'].unique()
)

#impact level filter
selected_impact = st.sidebar.multiselect(
    'Impact Level',
    options=['Low', 'Medium', 'High'],
    default=['Low', 'Medium', 'High']
)

#create new df to avoid changing original df_clean
df_filtered = df_clean.copy()

#date filter
df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])
df_filtered = df_filtered[(df_filtered['Date'] >= pd.to_datetime(start_date)) &
                          (df_filtered['Date'] <= pd.to_datetime(end_date))]

#sentiment filter
df_filtered = df_filtered[df_filtered['Sentiment'].isin(selected_sentiments)]

#sector filter
df_filtered = df_filtered[df_filtered['Sector'].isin(selected_sectors)]

#impact level filter
if selected_impact:
    df_filtered = df_filtered[df_filtered['Impact_Level'].isin(selected_impact)]

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Overview', 'Distributions', 'Categorical Analysis', 'Market Movement analysis', 'Correlations', 'Headline Text Analysis', 'Predictive Modelling'])

with tab1:
    overview.render(df, df_filtered, df_clean)
    
with tab2:
   data_distributions.render(df_filtered)

with tab3:
    categorical_analysis.render(df, df_filtered)
    
with tab4:
    market_movement_analysis.render(df_filtered)
    
with tab5:
    correlations.render(heatmap_df)

with tab6:
    nltk.render(clean_words, word_series)
        
with tab7:
    predictive.render(df_one_hot_encoded)
    




    


