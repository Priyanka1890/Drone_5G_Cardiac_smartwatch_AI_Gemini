import streamlit as st
from pydantic import BaseModel
#from pages import FitbitUser, Alarm, DroneComputer
from login_register_utils.logins import LoginData
from streamlit_js_eval import get_geolocation

import pandas as pd

class UserCredentials(BaseModel):
    username: str
    password: str
    role: str

def authenticate_user(credentials: UserCredentials, login: LoginData):
    df = login.check_login(usr=credentials.username, pwd=credentials.password, role=credentials.role)
    if df is not None and len(df) > 0:  # ensure df is not None before checking its length
        return df
    else:
        return None


def set_user_session(df:pd.DataFrame):
    if 'user_data' not in st.session_state:
        temp = {x:df[x].tolist()[0] for x in df.columns}
        st.session_state['user_data'] = temp


if __name__ == "__main__":

    st.set_page_config(
        page_title="Med4PAN",
        initial_sidebar_state="collapsed",
        page_icon="👋",
    )
    st.markdown(
        """
    <style>
        [data-testid="collapsedControl"] {
            display: none
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.title("Med4PAN Cardio Drone Service")
    st.markdown("Drone Delivery Service for Cardiac Emergency")

    login_data = LoginData("./data/register.csv")

    col1, col2 = st.columns(2)

    with col1:
        image = "https://static.vecteezy.com/system/resources/previews/025/903/589/original/medical-drone-and-and-medical-delivery-icon-concept-vector.jpg"
        st.image(image, caption='Cardiac abnormalities detecting using smartwatch, drone, defibrillator over 5G', use_column_width=True)

    with col2:
        role = st.selectbox(
            "Role",
            ("User", "Medical Professional", "Drone Operator")
        )
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        with st.container():
            col3, col4 = st.columns(2)
            with col3:
                if st.button("Login", use_container_width=True):
                    credentials = UserCredentials(username=username, password=password, role=role)
                    df = authenticate_user(credentials, login_data)
                    if df is not None:  # additional None check here
                        # now we know df is not None and we can safely access its properties
                        if df.role.tolist()[0] == "User":
                            set_user_session(df=df)
                            st.switch_page("pages/FitbitUser.py")
                        elif df.role.tolist()[0] == "Medical Professional":
                            set_user_session(df=df)
                            st.switch_page("pages/Alarm.py")
                        elif df.role.tolist()[0] == "Drone Operator":
                            set_user_session(df=df)
                            st.switch_page("pages/DroneComputer.py")
                    else:
                        st.error("Invalid username or password. Please try again.")

            with col4:
                if st.button("Register", use_container_width=True):
                    st.switch_page("pages/Signup.py")