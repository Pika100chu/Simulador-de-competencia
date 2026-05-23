
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

st.set_page_config(page_title="Simulador Oligopolio", layout="wide")

st.title("Simulador de Oligopolio")
st.write("Modelos disponibles: Cournot, Bertrand y Stackelberg")

# =====================
# FUNCIONES ECONÓMICAS
# =====================

def cournot(a,b,c1,c2):

    def precio(q1,q2):
        return a - b*(q1+q2)

    def sistema(vars):
        q1,q2 = vars
        eq1 = a-c1-2*b*q1-b*q2
        eq2 = a-c2-2*b*q2-b*q1
        return [eq1,eq2]

    q1_star,q2_star = fsolve(sistema,[10,10])

    Q_star = q1_star+q2_star
    P_star = precio(q1_star,q2_star)

    pi1 = (P_star-c1)*q1_star
    pi2 = (P_star-c2)*q2_star

    return q1_star,q2_star,Q_star,P_star,pi1,pi2


def stackelberg(a,b,c1,c2):

    def BR2(q1):
        return max((a-c2-b*q1)/(2*b),0)

    def beneficio_lider(q1):
        q2 = BR2(q1)
        P = a-b*(q1+q2)
        return (P-c1)*q1

    q1_grid = np.linspace(0,a/b,1000)
    beneficios = [beneficio_lider(q1) for q1 in q1_grid]

    q1_star = q1_grid[np.argmax(beneficios)]
    q2_star = BR2(q1_star)

    Q_star = q1_star+q2_star
    P_star = a-b*Q_star

    pi1 = (P_star-c1)*q1_star
    pi2 = (P_star-c2)*q2_star

    return q1_star,q2_star,Q_star,P_star,pi1,pi2


def bertrand(a,b,c1,c2):

    if c1 == c2:
        P_star = c1
        Q_star = (a-P_star)/b
        q1 = Q_star/2
        q2 = Q_star/2
        pi1 = 0
        pi2 = 0

    elif c1 < c2:
        P_star = c2
        Q_star = (a-P_star)/b
        q1 = Q_star
        q2 = 0
        pi1 = (P_star-c1)*q1
        pi2 = 0

    else:
        P_star = c1
        Q_star = (a-P_star)/b
        q1 = 0
        q2 = Q_star
        pi1 = 0
        pi2 = (P_star-c2)*q2

    return q1,q2,Q_star,P_star,pi1,pi2


# =====================
# GRÁFICOS
# =====================


def grafico_mercado(a,b,Q_star,P_star,titulo):
    Q = np.linspace(0,a/b,200)
    P_demanda = a-b*Q

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(Q,P_demanda,label="Demanda")
    ax.scatter(Q_star,P_star,s=250,zorder=3)

    # Líneas punteadas hasta los ejes
    ax.plot([Q_star,Q_star],[0,P_star], linestyle="--")
    ax.plot([0,Q_star],[P_star,P_star], linestyle="--")

    ax.set_xlabel("Cantidad total Q")
    ax.set_ylabel("Precio P")
    ax.set_title(titulo)
    ax.grid()
    ax.legend()
    return fig



def grafico_cournot(a,b,c1,c2,q1_star,q2_star):
    q = np.linspace(0,a,300)

    BR1 = (a-c1-q)/(2*b)
    BR2 = (a-c2-q)/(2*b)

    BR1 = np.maximum(BR1,0)
    BR2 = np.maximum(BR2,0)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(q,BR1,label='BR Empresa 1')
    ax.plot(BR2,q,label='BR Empresa 2')
    ax.scatter(q2_star,q1_star,s=250,zorder=3)

    # Líneas punteadas hasta ejes
    ax.plot([q2_star,q2_star],[0,q1_star], linestyle='--')
    ax.plot([0,q2_star],[q1_star,q1_star], linestyle='--')

    ax.set_xlabel('q2')
    ax.set_ylabel('q1')
    ax.set_title('Cournot - Curvas de reacción')
    ax.legend()
    ax.grid()
    return fig



def grafico_stackelberg(a,b,c2,q1_star,q2_star):
    q1 = np.linspace(0,a/b,300)
    BR2 = (a-c2-b*q1)/(2*b)
    BR2 = np.maximum(BR2,0)

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(q1,BR2,label='BR Seguidor')
    ax.scatter(q1_star,q2_star,s=250,zorder=3)

    # Líneas punteadas hasta los ejes
    ax.plot([q1_star,q1_star],[0,q2_star], linestyle='--')
    ax.plot([0,q1_star],[q2_star,q2_star], linestyle='--')

    ax.set_xlabel('q1 líder')
    ax.set_ylabel('q2 seguidor')
    ax.set_title('Stackelberg')
    ax.grid()
    ax.legend()
    return fig


# =====================
# INTERFAZ
# =====================

modelo = st.selectbox(
    "Seleccione modelo",
    ["Cournot","Bertrand","Stackelberg"]
)

col1,col2 = st.columns(2)

with col1:
    a = st.number_input("a (intercepto demanda)", min_value=0.01, value=100.0)
    b = st.number_input("b (pendiente demanda)", min_value=0.01, value=1.0)

with col2:
    c1 = st.number_input("Costo marginal empresa 1", value=10.0)
    c2 = st.number_input("Costo marginal empresa 2", value=20.0)

if st.button("Calcular equilibrio"):

    if modelo == "Cournot":
        resultado = cournot(a,b,c1,c2)

    elif modelo == "Bertrand":
        resultado = bertrand(a,b,c1,c2)

    else:
        resultado = stackelberg(a,b,c1,c2)

    q1_star,q2_star,Q_star,P_star,pi1,pi2 = resultado

    st.subheader(f"Equilibrio {modelo}")

    c1r,c2r,c3r = st.columns(3)

    c1r.metric("q1", f"{q1_star:.2f}")
    c1r.metric("q2", f"{q2_star:.2f}")

    c2r.metric("Q", f"{Q_star:.2f}")
    c2r.metric("P", f"{P_star:.2f}")

    c3r.metric("π1", f"{pi1:.2f}")
    c3r.metric("π2", f"{pi2:.2f}")

    if modelo == "Cournot":
        st.pyplot(grafico_cournot(a,b,c1,c2,q1_star,q2_star))
        st.pyplot(grafico_mercado(a,b,Q_star,P_star,"Mercado Cournot"))

    elif modelo == "Bertrand":
        st.pyplot(grafico_mercado(a,b,Q_star,P_star,"Mercado Bertrand"))

    else:
        st.pyplot(grafico_stackelberg(a,b,c2,q1_star,q2_star))
        st.pyplot(grafico_mercado(a,b,Q_star,P_star,"Mercado Stackelberg"))

    st.subheader("Comparación de modelos")

    cournot_res = cournot(a,b,c1,c2)
    bertrand_res = bertrand(a,b,c1,c2)
    stack_res = stackelberg(a,b,c1,c2)

    comparacion = pd.DataFrame({
        "Modelo":["Cournot","Bertrand","Stackelberg"],
        "Precio":[cournot_res[3], bertrand_res[3], stack_res[3]],
        "Cantidad":[cournot_res[2], bertrand_res[2], stack_res[2]],
        "Beneficio 1":[cournot_res[4], bertrand_res[4], stack_res[4]],
        "Beneficio 2":[cournot_res[5], bertrand_res[5], stack_res[5]]
    })

    st.dataframe(comparacion, use_container_width=True)

    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(comparacion["Modelo"], comparacion["Precio"])
    ax.set_title("Comparación de precios")
    ax.set_ylabel("Precio")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(5,4))
    ax2.bar(comparacion["Modelo"], comparacion["Cantidad"])
    ax2.set_title("Producción total")
    ax2.set_ylabel("Cantidad")
    st.pyplot(fig2)
