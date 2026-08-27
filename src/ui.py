"""Shared UI polish - call ui.inject() once near the top of each page."""
import streamlit as st

_CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
h1, h2, h3 {letter-spacing: -0.01em;}
[data-testid="stSidebar"] {background: #0B0E14;}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px; border-color: #232b3a;
}
.stButton > button {border-radius: 10px;}
[data-testid="stChatMessage"] {border-radius: 14px;}
[data-testid="stDataFrame"] {border-radius: 10px; overflow: hidden;}
[data-testid="stFileUploaderDropzone"] {border-radius: 12px;}
</style>
"""


def inject():
    st.markdown(_CSS, unsafe_allow_html=True)