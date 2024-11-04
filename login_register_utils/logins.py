
import pandas as pd
import os



class LoginData:
    def __init__(self, data_path:str):
        self.df = pd.read_csv(data_path)
    #def check_login(self, usr, pwd, role):
     #   return self.df[(self.df.username==usr) & (self.df.password == pwd) & (self.df.role == role)]
    

    def check_login(self, usr, pwd, role):
        user_df = self.df[(self.df.username == usr) & (self.df.role == role)]
        if user_df.empty:
            return None  # Return None if no user is found
        #saved_password = user_df['password'].values[0]
        #hashed = saved_password.encode()  # convert saved password hash
        if len(user_df)>0:# bcrypt.checkpw(pwd.encode(), hashed):
            return user_df  # return the dataframe if password is correct
        else:
            return None  # return None if password isn't correct
