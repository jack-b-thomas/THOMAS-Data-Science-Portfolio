import pandas as pd 
import streamlit as st 
from streamlit_option_menu import option_menu
import plotly_express as px
page = option_menu(
    menu_title = None,
    options= ['The Trade', 'Stats', 'Eye Test'],
    icons=['emoji-surprise', 'bar-chart', 'eye'],
    default_index=0,
    orientation='horizontal')
if page == 'The Trade': 
    st.html("<h1 style= 'text-allign: center; border-radius: 10px; color:#ff2b2b; font-size:40px'> It all went down on February 2, 2025 </h1>")
    st.html('<p> On February 2, 2025, just after midnight, the Mavericks traded Luka Doncic for Anthony Davis. The trade was unlike anything anyone had ever seen before. Luka Doncic is universally regarded as a top 5 player in the world. At only 25 years old, he has made All-NBA First Team five times. For context, Steph Curry has made All-NBA First Team four times. Last June, Luka played through several injuries and took the Mavs to the NBA Finals. Anthony Davis is by no means a bad player, but he is almost 32 years old, and has never carried a team like Luka has. His lone championship came with Lebron James in the bubble. So, why would the Mavs deal their franchise player?')
    st.html("<p> There are several theories that range from Luka's lack of conditioning to a politcal pressure scheme to legalize casinos in Texas. We may never know why Luka was traded, because the negotiations were so secretive. But here, you can compare Luka and AD's stats and conduct your own eye test to see if you have dealt Luka. (Spoiler Alert -- you wouldn't have)")
    st.header('The Complete Terms:')
    logo1, logo2, logo3 = st.columns([1,1,1], gap = 'large', vertical_alignment='top')
    with logo1: 
        st.image('https://cdn.nba.com/logos/nba/1610612747/primary/L/logo.svg', use_column_width=True) 
    with logo2:
        st.image('https://cdn.nba.com/logos/nba/1610612742/primary/L/logo.svg', use_column_width=True)
    with logo3:
        st.image('https://cdn.nba.com/logos/nba/1610612762/primary/L/logo.svg', use_column_width=True)
    col1, col2, col3 = st.columns([1,1,1], gap = 'large', vertical_alignment='top')
    with col1: 
        st.image('https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/3945274.png&w=350&h=254', use_column_width=True)
        st.html('<h5> Luka Doncic </h5>')
        st.image('https://cdn.nba.com/headshots/nba/latest/1040x760/1628467.png', use_column_width=True)
        st.html('<h5> Maxi Kleber </h5>')
        st.image('https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/6461.png&w=350&h=254', use_column_width=True)
        st.html('<h5> Markieff Morris')
    with col2: 
        st.image('https://www.mavs.com/wp-content/uploads/2025/02/AD.png')
        st.html('<h5> Anthony Davis </h5>')
        st.image('https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/4432582.png&w=350&h=254')
        st.html('<h5> Max Christie </h5>')
        st.image('https://d1yjjnpx0p53s8.cloudfront.net/styles/logo-thumbnail/s3/032018/untitled-3_26.png?gdrsLh8gefGS0SBTga7JCr1nQaK41g9w&itok=kOiJNVqI', use_column_width=True)
        st.html('<h5> 2029 First Round Pick </h5>') 
    with col3: 
        st.image('https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/5105797.png&w=350&h=254')
        st.html('<h5> Jalen Hood-Schifino </h5>')
        st.image('https://d1yjjnpx0p53s8.cloudfront.net/styles/logo-thumbnail/s3/032018/untitled-3_26.png?gdrsLh8gefGS0SBTga7JCr1nQaK41g9w&itok=kOiJNVqI', use_column_width=True)
        st.html('<h5> Two 2025 Second Round Picks </h5>')
elif page == 'Stats': 
    st.html("<h1 style= 'text-allign: center; border-radius: 10px; color:#ff2b2b; font-size:35px'> Player Statistics </h1>")
    st.write("Compare Luka and AD's statistics from the 2024-25 season, and see if the trade makes sense to you!")
    #df = pd.read_csv('portfolio_projects/luka_ad_trade_app/data/Luka_AD.csv')
    NBA_df = pd.read_csv('data/NBA Stats 202425 All Metrics  NBA Player Props Tool.csv')
    luka_df = NBA_df[NBA_df['NAME']== 'Luka Doncic']
    ad_df = NBA_df[NBA_df['NAME']== 'Anthony Davis']
    column1, column2 = st.columns([2,2], gap = 'medium', vertical_alignment='center') 
    with column1: 
        st.html('<h4> Luka Doncic </h4>')
        st.image('https://a.espncdn.com/combiner/i?img=/i/headshots/nba/players/full/3945274.png&w=350&h=254', use_column_width=True)
        st.dataframe(luka_df)
    with column2:
        st.html('<h4> Anthony Davis </h4>') 
        st.image('https://www.mavs.com/wp-content/uploads/2025/02/AD.png', use_column_width=True)
        st.dataframe(ad_df)
    choice = st.selectbox("Choose a Stat to Compare:", ['GP', 
                                               'AGE', 
                                               'MPG',
                                               'FT%', 
                                               '2P%', 
                                               '3P%', 
                                               'eFG%',
                                               'TS%', 
                                               'PpG',
                                               'RpG', 
                                               'ApG', 
                                               'SpG', 
                                               'BpG', 
                                               'TOpG', 
                                               'ORtg', 
                                               'DRtg'], 
                          index = 8)
    luka_choice = float(luka_df[choice])
    ad_choice = float(ad_df[choice])
    if choice == 'AGE': 
        st.html(f"Luka is <b> {luka_choice} </b> years old, whearas AD is <b> {ad_choice} </b> years old.")
    elif choice == 'GP': 
        st.html(f"Luka has played <b> {luka_choice} </b> games this season, whearas AD has played <b> {ad_choice} </b>this season.")
    elif choice == 'MPG': 
        st.html(f"Luka averages <b> {luka_choice} </b> minutes per game, whearas AD averages <b> {ad_choice} </b> minutes per game.")
    elif choice == 'FT%': 
        st.html(f"Luka's free throw percentage is <b> {luka_choice}</b>, whearas AD's is <b> {ad_choice}</b>.")
    elif choice == '2P%': 
        st.html(f"Luka's two point shooting percentage is <b>{luka_choice}</b>, whearas AD's is <b>{ad_choice}</b>.")
    elif choice == '3P%':
         st.html(f"Luka's three point shooting percentage is <b>{luka_choice}</b>, whearas AD's is <b>{ad_choice}</b>.")
    elif choice == 'eFG%':
         st.html(f"Luka's effective field goal percentage is <b>{luka_choice}</b>, whearas AD's is <b>{ad_choice}</b>.")
    elif choice == 'TS%': 
         st.html(f"Luka's true shooting percentage is <b>{luka_choice}</b>, whearas AD's is <b>{ad_choice}</b>.")
    elif choice == 'PpG': 
        st.html(f"Luka averages <b>{luka_choice}</b> points per game, whearas AD averages <b>{ad_choice}</b> points per game.")
    elif choice == "RpG": 
        st.html(f"Luka averages <b>{luka_choice}</b> rebounds per game, whearas AD averages <b>{ad_choice}</b> rebounds per game.")
    elif choice == "ApG": 
        st.html(f"Luka averages <b>{luka_choice}</b> assists per game, whearas AD averages <b>{ad_choice}</b> assists per game.")
    elif choice == 'SpG': 
        st.html(f"Luka averages <b>{luka_choice}</b> steals per game, whearas AD averages <b>{ad_choice}</b> steals per game.")
    elif choice == 'BpG':
        st.html(f"Luka averages <b>{luka_choice}</b> blocks per game, whearas AD averages <b>{ad_choice}</b> blocks per game.")
    elif choice == 'TOpG': 
        st.html(f"Luka averages <b>{luka_choice}</b> turnovers per game, whearas AD averages <b>{ad_choice}</b> turnovers per game.")
    elif choice == 'ORtg': 
        st.html(f"Luka's offensive rating is <b>{luka_choice}</b>, whearas AD's is <b>{ad_choice}</b>.")
    else: 
        st.html(f"Luka's defensive rating is <b>{luka_choice}</b>, whearas AD's defensive rating is <b>{ad_choice}</b>.")
    small_NBA_df = NBA_df.head(20)
    stat_fig = px.bar(small_NBA_df, x='NAME', y=choice, 
                      labels={choice:'', 'NAME': ''},
                      color='NAME',
                      color_discrete_sequence=['#FFCCCB','#FFCCCB','#FFCCCB','#FFCCCB','#ff2b2b','#FFCCCB','#FFCCCB','#FFCCCB','#FFCCCB','#FFCCCB','#FFCCCB','#ff2b2b','#FFCCCB','#FFCCCB','#FFCCCB','#FFCCCB'],
                      hover_name=choice,
                      title=f'{choice} Comparison for the Top 20 Players in PpG'
    )
    stat_fig.update_traces(textfont_size=10,
                           textposition='outside',
                           showlegend=False)
    stat_fig.update_layout(xaxis_tickangle= 45)
    st.plotly_chart(stat_fig) 
    st.html("<p style= 'text-alling: center; color: #D3D3D3; font-size:8px'> Player Data comes from <a href ='https://www.nbastuffer.com'> NBA Stuffer</a>")
else: 
    st.html("<h1 style= 'text-allign: center; border-radius: 10px; color:#ff2b2b; font-size:35px'> The Eye Test </h1>")
    st.write("Watch Luka and AD's highlight reels and see if this trade begins to make sense.")
    st.video('https://www.youtube.com/watch?v=QhU-cwIYL0w')
    st.video('https://www.youtube.com/watch?v=_-EK8G6GVVg')