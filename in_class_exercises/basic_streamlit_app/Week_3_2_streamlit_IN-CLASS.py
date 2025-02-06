# Import the Streamlit library
import streamlit as st

#use different commands to navigate around the terminal 
#ls lists everything within the directory  being used 
#cd navigates to a specific folder 
#streamlit run file: deploys a streamlit app
 
#these are the commands I used to open my streamlit app - just in case I forget 
#(base) jackthomas@dhcp-10-5-23-35 THOMAS-Data-Science-Portfolio % ls
#(base) jackthomas@dhcp-10-5-23-35 THOMAS-Data-Science-Portfolio % cd basic_streamlit_app
#(base) jackthomas@dhcp-10-5-23-35 basic_streamlit_app % streamlit run Week_3_2_streamlit_IN-CLASS.py 

# Display a simple text message

# Display a large title on the app

# ------------------------
# INTERACTIVE BUTTON
# ------------------------

# Create a button that users can click.
# If the button is clicked, the message changes.
st.html("<h2> Do you like Deadpool, Wolverine, or Jesus the most?</h2>")

if st.button("Deadpool"): 
    st.html('<h3> Awwww, thank you!!! <3 </h3>')
    st.image('https://pngimg.com/d/deadpool_PNG70.png')
else: 
    st.html("")
    
if st.button('Wolverine'): 
    st.html("<h3> Thanks, bub </h3>")
    st.image("https://images.bauerhosting.com/empire/2024/05/WOLV-CLAWS-TW.png?ar=16%3A9&fit=crop&crop=top&auto=format&w=1440&q=80")
else: 
    st.html('')

if st.button("Jesus"): 
    st.html("<h3> I am the Alpha and the Omega </h3>")
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQrminxLDznOV-2y1wepRStL3yphvcCz2L2CA&s")
else: 
    st.html('')
# ------------------------
# COLOR PICKER WIDGET
# ------------------------

# Creates an interactive color picker where users can choose a color.
# The selected color is stored in the variable 'color'.

# Display the chosen color value

# ------------------------
# ADDING DATA TO STREAMLIT
# ------------------------

# Import pandas for handling tabular data

# Display a section title

# Create a simple Pandas DataFrame with sample data


# Display a descriptive message

# Display the dataframe in an interactive table.
# Users can scroll and sort the data within the table.

# ------------------------
# INTERACTIVE DATA FILTERING
# ------------------------

# Create a dropdown (selectbox) for filtering the DataFrame by city.
# The user selects a city from the unique values in the "City" column.

# Create a filtered DataFrame that only includes rows matching the selected city.

# Display the filtered results with an appropriate heading.
  # Show the filtered table

# ------------------------
# NEXT STEPS & CHALLENGE
# ------------------------

# Play around with more Streamlit widgets or elements by checking the documentation:
# https://docs.streamlit.io/develop/api-reference
# Use the cheat sheet for quick reference:
# https://cheat-sheet.streamlit.app/

### Challenge:
# 1️⃣ Modify the dataframe (add new columns or different data).
# 2️⃣ Add an input box for users to type names and filter results.
# 3️⃣ Make a simple chart using st.bar_chart().


