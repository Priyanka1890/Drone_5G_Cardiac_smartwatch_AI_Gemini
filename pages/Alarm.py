import os.path
import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime
from fitbit_uitls.fitbit_funcs import get_last_min_heart_data, get_live_heart_rate
from streamlit_js_eval import get_geolocation


    

def main():


    st.title("Medical Monitor")
    usr_data = None
    if 'user_data' in st.session_state and st.session_state['user_data'] != "INVALID":
        usr_data = st.session_state['user_data']
    else:
        st.switch_page("home.py")


    tab1, tab2, tab3, tab4 = st.tabs([f"{usr_data['username']} Safe heart rate", "Contact Medical Professional", "Data from users", "Logout"])

    with tab1:
        st.header("Heart Rate Alarm !!!")
        values = st.slider(
            'Select a Safe Range for Heart Rate',
            0.0, 200.0, (60.0, 100.0))
        st.write('Values:', values)



        with open("alarm.json", 'w') as f:
            json.dump({"min":values[0], "max":values[1]}, f)

    
    with tab2:
        st.title("Video Conference")
        st.markdown("Join Video Conference [Telko Live](https://telko.live/5bc36d47b87dad6c)")


    with tab3:

        st.markdown("Get data using Machine learning models")
    with tab4:
        if st.button("Logout"):
            st.session_state['user_data'] = "INVALID"
            st.switch_page("home.py")


   


if __name__ == '__main__':
    main()