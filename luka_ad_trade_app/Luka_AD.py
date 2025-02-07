import pandas as pd 
import streamlit as st 
from streamlit_option_menu import option_menu
page = option_menu(
    menu_title = None,
    options= ['The Trade', 'Stats', 'Eye Test'],
    icons=['emoji-surprise', 'bar-chart', 'eye'],
    default_index=0,
    orientation='horizontal')

if page == 'The Trade': 
    st.markdown("<h1 style= 'text-allign: center; border-radius: 10px; color:#ff2b2b; font-size:24p'> It all went down on February 2, 2025 </h1>", unsafe_allow_html=True)
    st.header('The Terms:')
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
    st.markdown("<h1 style= 'text-allign: center; border-radius: 10px; color:#ff2b2b; font-size:20p'> Player Statistics </h1>", unsafe_allow_html=True)
    st.write("Compare Luka and AD's statistics from the 2024-25 season, and see if the trade makes sense to you!")
    df = pd.read_csv('Luka_AD.csv')
    NBA_df = pd.read_csv('NBA Stats 202425 All Metrics  NBA Player Props Tool.csv')
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
                                               'TpG', 
                                               'ORtg', 
                                               'DRtg'], 
                          index = 8)
    luka_choice = float(luka_df[choice])
    ad_choice = float(ad_df[choice])
    if choice == 'AGE': 
        st.write(f"Luka is {luka_choice} years old, whearas AD is {ad_choice} years old.")
    elif choice == 'GP': 
        st.write(f"Luka has played {luka_choice} games this season, whearas AD has played {ad_choice} this season.")
    elif choice == 'MPG': 
        st.write(f"Luka averages {luka_choice} minutes per game, whearas AD averages {ad_choice} minutes per game.")
    elif choice == 'FT%': 
        st.write(f"Luka's free throw percentage is {luka_choice}, whearas AD's is {ad_choice}.")
    elif choice == '2P%': 
        st.write(f"Luka's two point shooting percentage is {luka_choice}, whearas AD's is {ad_choice}.")
    elif choice == '3P%':
         st.write(f"Luka's three point shooting percentage is {luka_choice}, whearas AD's is {ad_choice}.")
    elif choice == 'eFG%':
         st.write(f"Luka's effective field goal percentage is {luka_choice}, whearas AD's is {ad_choice}.")
    elif choice == 'TS%': 
         st.write(f"Luka's true shooting percentage is {luka_choice}, whearas AD's is {ad_choice}.")
    elif choice == 'PpG': 
        st.write(f"Luka averages {luka_choice} points per game, whearas AD averages {ad_choice} points per game.")
    elif choice == "RpG": 
        st.write(f"Luka averages {luka_choice} rebounds per game, whearas AD averages {ad_choice} rebounds per game.")
    elif choice == "ApG": 
        st.write(f"Luka averages {luka_choice} assists per game, whearas AD averages {ad_choice} assists per game.")
    elif choice == 'SpG': 
        st.write(f"Luka averages {luka_choice} steals per game, whearas AD averages {ad_choice} steals per game.")
    elif choice == 'BpG':
        st.write(f"Luka averages {luka_choice} blocks per game, whearas AD averages {ad_choice} blocks per game.")
    elif choice == 'TpG': 
        st.write(f"Luka averages {luka_choice} turnovers per game, whearas AD averages {ad_choice} turnovers per game.")
    elif choice == 'ORtg': 
        st.write(f"Luka's offensive rating is {luka_choice}, whearas AD's is {ad_choice}.")
    else: 
        st.write(f"Luka's defensive rating is {luka_choice}, whearas AD's defensive rating is {ad_choice}")
    st.write('Do any of these stats help you make sense of the trade?')
else: 
    st.markdown("<h1 style= 'text-allign: center; border-radius: 10px; color:#ff2b2b; font-size:20p'> The Eye Test </h1>", unsafe_allow_html=True)
    st.write("Watch Luka and AD's highlight reels and see if this trade begins to make sense.")
    st.video('https://www.youtube.com/watch?v=QhU-cwIYL0w')
    st.video('https://www.youtube.com/watch?v=_-EK8G6GVVg')