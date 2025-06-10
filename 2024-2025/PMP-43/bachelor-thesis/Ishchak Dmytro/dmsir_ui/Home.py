import streamlit as st, pandas as pd
from dmsir_core.dmsir_dde import simulate, Params

st.set_page_config("D-MSIR demo", layout="centered")
st.title("D-MSIR · COVID-19 (демо)")

beta = st.slider("β", .1, .5, .28, .01)
tau  = st.slider("τ, доби", 1., 7., 3.7, .1)
days = st.slider("Горизонт, д", 60, 360, 180, 10)

df = pd.DataFrame(simulate(days, .25, Params(beta=beta, tau=tau)),
                  columns="t D M I R".split())
st.line_chart(df.set_index("t")[["I", "D", "M", "R"]])
