import streamlit as st
from ejercicios import get_ejercicios

from ejercicios.ejercicio_1 import ejercicio_1
from ejercicios.ejercicio_2 import ejercicio_2
from ejercicios.ejercicio_3 import ejercicio_3
from ejercicios.ejercicio_4 import ejercicio_4
from ejercicios.ejercicio_5 import ejercicio_5
from ejercicios.ejercicio_6 import ejercicio_6
from ejercicios.ejercicio_7 import ejercicio_7
from ejercicios.ejercicio_8 import ejercicio_8
from ejercicios.ejercicio_9 import ejercicio_9
from ejercicios.ejercicio_10 import ejercicio_10
from ejercicios.ejercicio_11 import ejercicio_11

st.title("Menú de Ejercicios de Visión Artificial")

ejercicios = get_ejercicios()
ejercicio_seleccionado = st.selectbox("Selecciona un ejercicio:", ejercicios)

if ejercicio_seleccionado == "Ejercicio 1":
    ejercicio_1()
elif ejercicio_seleccionado == "Ejercicio 2":
    ejercicio_2()
elif ejercicio_seleccionado == "Ejercicio 3":
    ejercicio_3()
elif ejercicio_seleccionado == "Ejercicio 4":
    ejercicio_4()
elif ejercicio_seleccionado == "Ejercicio 5":
    ejercicio_5()
elif ejercicio_seleccionado == "Ejercicio 6":
    ejercicio_6()
elif ejercicio_seleccionado == "Ejercicio 7":
    ejercicio_7()
elif ejercicio_seleccionado == "Ejercicio 8":
    ejercicio_8()
elif ejercicio_seleccionado == "Ejercicio 9":
    ejercicio_9()
elif ejercicio_seleccionado == "Ejercicio 10":
    ejercicio_10()
elif ejercicio_seleccionado == "Ejercicio 11":
    ejercicio_11()
else:
    st.info("Selecciona el ejercicio 1, 2, 3, 4, 5, 6, 7, 8, 9 o 10 para ver su contenido. (Otros ejercicios próximamente)")
