import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def render(df_filtered):
    #seaborn boxplots exploring how index percent chane and trading volume are affected by various factors
    st.write('**How do market returns and trading activity differ across news-related categories?**')

    st.write(
        'This section examines how market outcomes, specifically index percentage changes and trading volumes vary across different'
        'contextual factors such as event impact level, news sentiment, and market sector. Boxplots are used to compare distributions,'
        'spreads, and outliers, allowing assessment of whether certain categories are associated with systematically different market'
        'behaviour.'
    )

    st.write(
        '**Note:** The visualisations in this tab reflect the currently selected date range. Adjusting the date filter may change'
          'the shape of these distributions, highlighting how news and market behaviour vary across different periods.'
    )

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Impact_Level', y='Index_Change_Percent', data=df_filtered, ax=ax, palette='Set2', linewidth=2)
    ax.set_title('Index Change Percent by Impact Level')
    ax.set_ylabel('Index Change Percent')
    ax.set_xlabel('Impact Level')
    ax.grid(True, axis='y')
    st.pyplot(fig)

    st.write(
        'This chart compares the distribution of index percentage changes across low, medium, and high impact events within the'
        'selected time period. Differences in medians, spreads, or outliers may suggest variation in market responses across'
        'impact levels.'
    )

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Sentiment', y='Index_Change_Percent', data=df_filtered, ax=ax, palette='Set3', linewidth=2)
    ax.set_title('Index Change Percent by Sentiment')
    ax.set_ylabel('Index Change Percent')
    ax.set_xlabel('Sentiment')
    ax.grid(True, axis='y')
    st.pyplot(fig)

    st.write(
        'This chart compares index percentage changes across positive, neutral, and negative sentiment categories. Overlap between'
        'distributions indicates uncertainty, while shifts in medians may suggest sentiment-related differences in market response.'
    )

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Sector', y='Index_Change_Percent', data=df_filtered, ax=ax, palette='pastel', linewidth=2)
    ax.set_title('Index Change Percent by Sector')
    ax.set_ylabel('Index Change Percent')
    ax.set_xlabel('Sector')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write('This chart displays index changes for events affecting different sectors, revealing which sectors experience '
             'greater volatility or larger shifts in response to news during the selected period.')

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Impact_Level', y='Trading_Volume', data=df_filtered, ax=ax, palette='Set2', linewidth=2)
    ax.set_title('Trading Volume by Impact Level')
    ax.set_ylabel('Trading Volume')
    ax.set_xlabel('Impact Level')
    ax.grid(True, axis='y')
    st.pyplot(fig)

    st.write(
        'This chart shows how trading volume varies across impact levels. Wider spreads or higher medians may suggest increased'
        'market activity following higher-impact events.'
    )

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Sentiment', y='Trading_Volume', data=df_filtered, ax=ax, palette='Set3', linewidth=2)
    ax.set_title('Trading Volume by Sentiment')
    ax.set_ylabel('trading Volume')
    ax.set_xlabel('Sentiment')
    ax.grid(True, axis='y')
    st.pyplot(fig)

    st.write(
        'This chart displays trading volume distributions across sentiment categories. Differences in spread or central tendency'
        'may reflect varying levels of market engagement.'
    )

    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Sector', y='Trading_Volume', data=df_filtered, ax=ax, palette='pastel', linewidth=2)
    ax.set_title('Trading Volume by Sector')
    ax.set_ylabel('Trading Volume')
    ax.set_xlabel('Sector')
    ax.grid(True, axis='y')
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.write(
        'This chart shows how trading volume varies across sectors. Outliers and dispersion provide insight into sector-specific'
        'responses to news.'
    )