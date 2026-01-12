import streamlit as st 

def render(df_filtered, df, df_clean):
    col1, col2, col3, col4, col5 = st.columns(5)
    #key metrics at a glance
    col1.metric('Total Headlines: ', len(df_filtered))
    col2.metric('Average Index Change Percent: ', f'{round(df_filtered["Index_Change_Percent"].mean(), 2)}%')
    col3.metric('Most Common Sentiment: ', df_filtered['Sentiment'].mode()[0])
    col4.metric('Most Impacted Sector: ', df_filtered['Sector'].mode()[0])
    col5.metric('Most Common Event Type: ', df_filtered['Market_Event'].mode()[0])

    st.write('Original Dataset Shape: ', df.shape)

    st.write('Missing Values Before Data cleaning:')
    missing_per_column = df.isna().sum().reset_index()
    missing_per_column.columns = ['Column', 'Missing Values']
    st.dataframe(missing_per_column)

    st.write('Rows missing Headline or Index_Change_Percent were removed. Missing sentiment scores were filled using sentiment'
            'analysis on the headlines, and missing URL values were replaced with \'missing url data\'.')
    
    st.write('Final Dataset Shape: ', df_clean.shape)
    
    st.write('Summary Statistics Post Data Cleaning: \n')
    st.write(df_clean.describe())

    st.write('Clean Dataset Preview: ')
    st.dataframe(df_clean.head())