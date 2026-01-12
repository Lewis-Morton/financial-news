import streamlit as st 
import seaborn as sns
import matplotlib.pyplot as plt

def render(heatmap_df):
    #seaborn heatmap which explores correlations between key features, encoding was done in jupyter and heatmap_df was exported
    st.write('**How are key numerical features related across the full dataset?**')
    st.write(
        'This heatmap visualizes the correlations between trading sentiment, trading volume, index change percent and impact level.'
        'Correlations are computed across the entire dataset to highlight overall linear relationships that persist beyond specific'
        'time windows.'
    )
    
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(heatmap_df.corr(), annot=True, cmap='RdBu_r', center=0)
    ax.set_title("Correlation Heatmap of Key Financial Features")
    st.pyplot(fig)

    st.write(
        'The relationships in the heatmap are generally weak. The strongest correlation is only 0.037, found between sentiment and'
        'trading volume, suggesting that sentiment in headlines does not strongly align with trading behavior in this dataset.'
    )