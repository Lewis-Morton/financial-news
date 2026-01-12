import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def render(df_filtered):
    st.write('**How are key news and market features distributed over time?**')
    st.write(
            'This section explores how news sentiment, trading activity, market movements, and event impact are distributed within'
            'the dataset. These distributions provide context on what types of events are most common and how market responses are'
            'typically spread.'
        )
    st.write(
            '**Note:** The visualisations in this tab reflect the currently selected date range. Adjusting the date filter may change'
            'the shape of these distributions, highlighting how news and market behaviour vary across different periods.'
        )

    fig, ax = plt.subplots(figsize=(12,6))
    sns.countplot(x='Sentiment', data=df_filtered, palette='Set2', edgecolor='black', linewidth=2, ax=ax)
    ax.set_title('Sentiment Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('News Sentiment')
    ax.grid(True, axis='y')
    st.pyplot(fig)
        
    st.write(
            'This chart shows how news headlines are distributed across positive, neutral, and negative sentiment categories within'
            'the selected time period. A skew toward a particular sentiment may indicate prevailing market narratives during that period.'
        )
    

    fig, ax = plt.subplots(figsize=(12, 6))
    df_filtered['Trading_Volume'].plot(kind='hist', bins=10, ax=ax, color='mistyrose', edgecolor='black', linewidth=2)
    ax.set_title('Trading Volume Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Trading Volume')
    ax.grid(True, axis='y')
    st.pyplot(fig)
        
    st.write(
            'The histogram illustrates how often different trading volume ranges occur. Concentration around certain values may indicate common'
            'trading levels while long tails could indicate unusual trading activity.'
            
        )


    fig, ax = plt.subplots(figsize=(12,6))
    df_filtered['Index_Change_Percent'].plot(kind='hist', bins=10, ax=ax, color='honeydew', edgecolor='black', linewidth=2)
    ax.set_title('Index Change % Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Index Change Percent')
    ax.grid(True, axis='y')
    st.pyplot(fig)
        
    st.write(
            'This chart displays the distribution of market returns following news events. The shape of the distribution provides'
            'insight into the frequency and magnitude of positive and negative market movements.'
        )


    fig, ax = plt.subplots(figsize=(12,6))
    sns.countplot(x='Impact_Level', data=df_filtered, palette='Set3', edgecolor='black', linewidth=2, ax=ax)
    ax.set_title('Impact Level Distribution')
    ax.set_ylabel('Count')
    ax.set_xlabel('Impact Level')
    ax.grid(True, axis='y')
    st.pyplot(fig)                    
        
    st.write(
            'This chart shows how event impact scores are distributed across news events. A concentration of low or high impact values'
            'can suggest whether most news events tend to have limited or substantial market influence.'
        )
