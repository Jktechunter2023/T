from src.airplane_poc import my_app
import streamlit as st
from src.sms_alert import send_sms
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

if __name__ == "__main__":

    #my_app(wide_layout=True)
    my_app()
