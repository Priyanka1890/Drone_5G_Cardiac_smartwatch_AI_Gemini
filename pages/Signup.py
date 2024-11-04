import streamlit as st
import pandas as pd
import os
import re
from datetime import date
import bcrypt


def hash_password(password: str) -> str:
    password = password.encode('utf-8')  # Passwort muss ein Byte-String sein
    salt = bcrypt.gensalt()  # Erstellt einen neuen Salt
    hashed_password = bcrypt.hashpw(password, salt)  # Hasht das Passwort
    return hashed_password.decode('utf-8')  # Kehrt als String zurück

def signup():
    st.title("Signup Page")
    username = st.text_input("Username")
    password_clear = st.text_input("Password", type="password")
    password = hash_password(password_clear)
    role = st.selectbox("Role", ("User", "Medical Professional", "Drone Operator"))
    firstname = st.text_input("First Name").capitalize()
    lastname = st.text_input("Last Name").capitalize()
    sex = st.selectbox("Sex", ("Male", "Female"))
    min_birthdate = date(1890, 1, 1)
    max_birthdate = date.today()
    default_date = date(1990, 1, 1)
    birthdate = st.date_input("Birthdate", default_date, min_value=min_birthdate, max_value=max_birthdate)
    email = st.text_input("Email")
    city = st.text_input("City")
    postalcode = st.text_input("Postal Code")
    street = st.text_input("Street Name and House Number")

    password_strong = is_password_strong(password_clear)
    email_valid = is_email_valid(email)
    postalcode_valid = is_postalcode_valid(postalcode)
    street_valid = is_street_valid(street)

    if not password_clear or not password_strong:
        st.warning("Password is too weak. It should contain at least 8 characters with a mix of uppercase, lowercase, "
                   "numbers, and _@$ characters.")
    elif not email or not email_valid:
        st.warning("The email address you have entered isn't valid")
    elif not postalcode or not postalcode_valid:
        st.warning("The Postal Code you have entered isn't valid")
    elif not street or not street_valid:
        st.warning("The Street you have entered isn't valid")

    signup_button_enabled = username and password and password_strong and firstname and lastname and birthdate and email and email_valid and city and postalcode and postalcode_valid and street
    signup_button = st.button("Signup", disabled=not signup_button_enabled)

    if signup_button:
        if os.path.exists("./data/register.csv"):
            df = pd.read_csv("./data/register.csv")
        else:
            df = pd.DataFrame(columns=['username', 'password', 'role', 'firstname', 'lastname', 'sex', 'birthdate', 'email', 'city', 'postalcode', 'street'])

        if df[df['username'] == username].shape[0] == 0:
            # Append new user data
            new_user = {'username': username, 'password': password, 'role': role, 'firstname': firstname,
                        'lastname': lastname, 'sex': sex, 'birthdate': birthdate, 'email': email, 'city': city,
                        'postalcode': postalcode, 'street': street}
            new_user_df = pd.DataFrame([new_user])
            df = pd.concat([df, new_user_df], ignore_index=True)
            df.to_csv("./data/register.csv", index=False)
            st.success("Successfully created a new account.")
            st.switch_page("pages/FitbitUser.py")
        else:
            st.error("This username is already taken. Please try a different username.")

def is_password_strong(password):
    """
    Function to check if the password is strong enough.
    """
    if len(password) < 8:
        return False
    elif not re.search("[a-z]", password):
        return False
    elif not re.search("[A-Z]", password):
        return False
    elif not re.search("[0-9]", password):
        return False
    elif not re.search("[_@$?!#,:;-_+äöü§%&/(){}]", password):
        return False
    elif re.search("\s", password):
        return False
    else:
        return True

def is_email_valid(email):
    pattern = r"[^@]+@[^@]+\.[^@]+"
    return re.match(pattern, email) is not None

def is_postalcode_valid(postalcode):
    pattern = r"\d{5}"
    return re.match(pattern, postalcode) is not None

def is_street_valid(street):
    pattern = r"^[a-zA-ZäöüÄÖÜß\.]+\s[1-9][0-9]{0,2}$"
    return re.match(pattern, street) is not None

def is_city_valid(city):
    pattern = r"^[a-zA-ZäöüÄÖÜß\s]+$"
    return re.match(pattern, city) is not None

if __name__ == "__main__":
    signup()