
# -*- coding: utf-8 -*-
"""
DSS DE INTEGRIDAD Y MANTENIMIENTO DE TUBERÍAS DE REFINERÍA
Interfaz Streamlit
===========================================================
Flujo:
Tipo de tubería -> carga de datos -> 7 capas -> KPI por tubería/tramo ->
políticas -> reporte Excel final.
"""

from __future__ import annotations
from pathlib import Path
from tempfile import NamedTemporaryFile
import io
import os
import sqlite3
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xml.etree.ElementTree as ET

import dss_drbi as de

# =============================================================================
# 0. CONFIGURACIÓN GENERAL Y ESTILOS
# =============================================================================

st.set_page_config(page_title="DSS Integridad de Tuberías", page_icon="🛢️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
.dss-title {background: linear-gradient(90deg,#0b1f3a,#17365d,#2e75b6); color:white; padding:22px 26px; border-radius:16px; margin-bottom:14px;}
.dss-title h1 {margin:0; font-size:2rem;}
.dss-title p {margin:.4rem 0 0 0; color:#dbe8f6;}
.panel {border:1px solid #d9e2f3; border-radius:14px; padding:14px 16px; background:#ffffff; margin-bottom:10px;}
.caption-box {background:#f5f8fc; border-left:4px solid #2e75b6; padding:10px 12px; border-radius:8px; color:#475467;}
div[data-testid="stMetric"] {border:1px solid #d9e2f3; background:white; padding:12px; border-radius:14px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dss-title">
  <h1>DSS – Integridad y Mantenimiento de Tuberías</h1>
  <p>IIoT/ETL → IOWs → Bayes → DRBI → MDP-RVO → MILP → Evaluación KPI</p>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# 1. EJECUCIÓN AUTOMÁTICA DEL MOTOR (SIN BLOQUEO)
# =============================================================================
@st.cache_data(show_spinner=False)
def get_cached_results(tipo_tuberia):
    # Usamos la data de demostración base para evitar que la app empiece vacía
    dataset = de.demo_dataset()
    # Ejecuta todas las capas automáticamente
    return de.run_all(dataset, tipo_tuberia, auto_generate_windows=True)

def ensure_state():
    if "tipo_tuberia" not in st.session_state:
        st.session_state.tipo_tuberia = de.TIPOS_TUBERIA[0]

ensure_state()

with st.spinner("Cargando motor analítico e inicializando datos..."):
    result = get_cached_results(st.session_state.tipo_tuberia)

df_c1 = result.get("capa1", pd.DataFrame())
df_c2 = result.get("capa2", pd.DataFrame())
df_c3 = result.get("capa3", pd.DataFrame())
df_c4 = result.get("capa4", pd.DataFrame())
df_c5 = result.get("intervenciones", pd.DataFrame())
df_c6 = result.get("programa", pd.DataFrame())
df_c7 = result.get("rolling", pd.DataFrame())

def show_data(df, info_text):
    if df is not None and not df.empty:
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info(info_text)

def money(v) -> str:
    try:
        return f"S/ {float(v):,.2f}"
    except:
        return "S/ 0.00"

def fmt_date(value) -> str:
    if pd.isna(value): return "—"
    try: return pd.Timestamp(value).strftime("%d/%m/%Y")
    except: return str(value)

# =============================================================================
# 2. PANEL LATERAL DE NAVEGACIÓN
# =============================================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cc/Petroper%C3%BA_Logo.svg/1200px-Petroper%C3%BA_Logo.svg.png", width=180)
    st.markdown("---")
    st.header("Configuración")
    
    nuevo_tipo = st.selectbox(
        "Filtrar Tipo de Tubería:",
        de.TIPOS_TUBERIA,
        index=de.TIPOS_TUBERIA.index(st.session_state.tipo_tuberia)
    )
    if nuevo_tipo != st.session_state.tipo_tuberia:
        st.session_state.tipo_tuberia = nuevo_tipo
        st.rerun()

    st.markdown("---")
    st.header("Fases del Modelo")
    menu = st.radio("Secciones:", [
        "📊 Dashboard Ejecutivo",
        "C1: Datos y Calidad",
        "C2: Excursiones IOW",
        "C3: Actualización Bayesiana",
        "C4: Riesgo DRBI",
        "C5: Decisión MDP-RVO",
        "C6: Planificación MILP",
        "C7: Evaluación KPI",
        "📥 Ingreso de Activos (XML)",
        "📜 Políticas Corporativas",
        "📄 Generador de Excel"
    ])
    st.divider()
    st.caption(f"Versión: DSS v{de.VERSION_DSS}")

# =============================================================================
# 3. PESTAÑAS Y VISTAS
# =============================================================================

if menu == "📊 Dashboard Ejecutivo":
    kpis = result.get("kpis", {})
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tramos Evaluados", kpis.get("Tramos evaluados", 0))
    c2.metric("Trabajos Programados", kpis.get("Trabajos programados", 0))
    c3.metric("Riesgo Promedio", f"{kpis.get('Riesgo DRBI promedio', 0):.4f}")
    c4.metric("Presupuesto Requerido", f"S/ {kpis.get('Presupuesto estimado de la cartera (S/)', 0):,.0f}")

    if not df_c4.empty:
        df_latest = df_c4.sort_values("Fecha_hora").groupby("ID_tramo").tail(1)
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig1 = px.scatter(df_latest, x="PoF_posterior", y="CoF_valor", color="NIVEL_RIESGO",
                              hover_data=["ID_tramo", "Riesgo_DRBI"],
                              title="Matriz de Riesgo Dinámico",
                              color_discrete_map={"BAJO":"#2e7d32", "MEDIO":"#d98e04", "ALTO":"#f57c00", "CRITICO":"#c62828"})
            st.plotly_chart(fig1, use_container_width=True)
            
        with col_chart2:
            fig2 = px.pie(df_latest, names="NIVEL_RIESGO", title="Distribución de Tramos por Riesgo",
                          color="NIVEL_RIESGO", color_discrete_map={"BAJO":"#2e7d32", "MEDIO":"#d98e04", "ALTO":"#f57c00", "CRITICO":"#c62828"})
            st.plotly_chart(fig2, use_container_width=True)

elif menu == "C1: Datos y Calidad":
    st.subheader("C1: Ingesta, Calidad y ETL (IIoT / DCS / LIMS)")
    st.info("Datos consolidados, limpios de duplicados e interpolados listos para el análisis.")
    show_data(de.table_layer1(result), "No hay datos procesados en la Capa 1.")

elif menu == "C2: Excursiones IOW":
    st.subheader("C2: Motor de Ventanas Operativas (IOW)")
    st.info("Cálculo de desviaciones operativas, considerando magnitud, duración y frecuencia.")
    show_data(de.table_layer2(result), "No hay excursiones calculadas en la Capa 2.")

elif menu == "C3: Actualización Bayesiana":
    st.subheader("C3: Actualización Bayesiana (BBN)")
    st.info("Actualización de la Probabilidad de Falla (PoF) integrando la evidencia IOW.")
    show_data(de.table_layer3(result), "No hay datos de la Capa 3.")

elif menu == "C4: Riesgo DRBI":
    st.subheader("C4: Riesgo DRBI")
    st.info("Cálculo del riesgo dinámico combinando la PoF Bayesiana con la Consecuencia de Falla (CoF).")
    show_data(de.table_layer4(result), "No hay datos de la Capa 4.")

elif menu == "C5: Decisión MDP-RVO":
    st.subheader("C5: Markov Decision Process (MDP) y RVO")
    st.info("Asignación de la acción prescriptiva óptima y la restricción de ventana operativa (RVO).")
    show_data(de.table_layer5(result), "No hay datos de la Capa 5.")

elif menu == "C6: Planificación MILP":
    st.subheader("C6: Programación Lineal Entera Mixta (MILP)")
    st.info("Asignación de fechas y recursos mediante SciPy/HiGHS respetando capacidades, presupuesto y ventanas operativas.")
    df_prog = de.table_layer6(result)
    
    if not df_prog.empty:
        col1, col2 = st.columns(2)
        col1.metric("Trabajos Programados MILP", len(df_prog[df_prog["Estado de programación"] == "PROGRAMADO"]))
        col2.metric("Trabajos Sin Asignar", len(df_prog[df_prog["Estado de programación"] != "PROGRAMADO"]))
        
    show_data(df_prog, "No se generó el programa MILP.")

elif menu == "C7: Evaluación KPI":
    st.subheader("C7: Evaluación de Desempeño y Rolling Horizon")
    st.info("Cálculo individual de KPIs por tramo y reevaluación del horizonte ante nueva evidencia.")
    
    st.markdown("#### Indicadores de Gestión de Integridad")
    show_data(de.table_layer7_kpis(result), "No hay KPIs generados.")
    
    st.markdown("#### Reprogramación (Rolling Horizon)")
    show_data(de.table_layer7(result), "No hay datos de reprogramación.")

elif menu == "📥 Ingreso de Activos (XML)":
    st.subheader("📥 Ingesta Masiva XML")
    st.info("Suba el archivo con formato XML para añadir más tramos a la plataforma.")
    
    xml_out = os.path.join(tempfile.gettempdir(), f"Plantilla_DSS_{st.session_state.tipo_tuberia}.xml")
    de.xml_template(xml_out, st.session_state.tipo_tuberia)
    with open(xml_out, "rb") as f:
        st.download_button("📥 Descargar Plantilla XML", f, file_name=f"Plantilla_{st.session_state.tipo_tuberia}.xml", mime="application/xml")
        
    uploaded_file = st.file_uploader("Sube el archivo XML de activos", type=["xml"])
    if uploaded_file is not None:
        try:
            tree = ET.parse(uploaded_file)
            root = tree.getroot()
            data = []
            for child in root:
                row = {subchild.tag: subchild.text for subchild in child}
                data.append(row)
            df_xml = pd.DataFrame(data)
            st.success("¡XML Procesado! Los datos se integrarán en la próxima corrida del ETL.")
            st.dataframe(df_xml, use_container_width=True)
        except Exception as e:
            st.error(f"Error procesando XML: {e}")

elif menu == "📜 Políticas Corporativas":
    st.subheader("📜 Políticas de Mantenimiento Prescriptivo")
    st.markdown("Directrices del modelo analítico para priorizar, presupuestar y ejecutar el mantenimiento.")
    
    with st.expander("Política General por Nivel de Riesgo", expanded=True):
        show_data(result.get("politicas", pd.DataFrame()), "No disponible.")
    with st.expander("Reglas de Programación y MILP", expanded=True):
        show_data(result.get("politica_programacion", pd.DataFrame()), "No disponible.")

elif menu == "📄 Generador de Excel":
    st.subheader("📄 Exportación de Reporte Final")
    st.info("Descarga el reporte corporativo consolidado en Excel, con dashboard dinámico por tramo.")
    
    if st.button("🚀 Compilar y Generar Excel", type="primary"):
        with st.spinner("Generando reporte..."):
            out_path = os.path.join(tempfile.gettempdir(), f"Reporte_DSS_{st.session_state.tipo_tuberia}.xlsx")
            report_file = de.export_excel(result, st.session_state.tipo_tuberia, out_path)
            
            with open(report_file, "rb") as f:
                st.download_button(
                    label="✅ Clic Aquí para Descargar",
                    data=f,
                    file_name=f"Reporte_DSS_{st.session_state.tipo_tuberia}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

