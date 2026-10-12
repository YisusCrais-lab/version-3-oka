# -*- coding: utf-8 -*-
"""
DSS DE INTEGRIDAD Y MANTENIMIENTO DE TUBERÍAS DE REFINERÍA
=============================================================
Motor de cálculo para el modelo integrado:
IIoT/ETL -> IOWs -> actualización bayesiana -> DRBI -> MDP-RVO -> MILP
-> evaluación de KPI y reprogramación por horizonte rodante.

OBJETIVO DE ESTA VERSIÓN
------------------------
1. Mantener la lógica matemática definida en el código base del proyecto.
2. Separar claramente cálculo, validación, presentación y exportación.
3. Trabajar con tres familias de tuberías: Submarina, Transferencia y Proceso.
4. Evitar dependencias frágiles del solver: el MILP se resuelve con SciPy/HiGHS,
   conservando la misma función objetivo y las mismas restricciones del modelo base.
5. Producir salidas comprensibles para usuarios de Operaciones, Integridad,
   Mantenimiento y Planificación.

NOTA DE IMPLEMENTACIÓN
----------------------
Los parámetros de demostración (límites IOW, costos, capacidades, ventanas,
matrices de transición y umbrales) son configurables. Antes de un uso operativo
se deben reemplazar por parámetros aprobados por la organización. Esta nota no
altera las ecuaciones del modelo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import html
import json
import math
import os
import re
import tempfile
import warnings

import numpy as np
import pandas as pd

# =============================================================================
# 0. IDENTIFICACIÓN DE VERSIÓN Y CATÁLOGOS GENERALES
# =============================================================================

VERSION_DSS = "2.4.0"
NOMBRE_MODELO = "DRBI–IOWs–Bayes–MDP-RVO–MILP"
TIPOS_TUBERIA = ["Submarina", "Transferencia", "Proceso"]

# Etiquetas legibles. Se usan en la interfaz y en el Excel; no intervienen en
# los cálculos.
GLOSARIO = {
    "IIoT": "Integración de datos operativos, laboratorio, inspección y mantenimiento.",
    "IOW": "Ventana Operativa de Integridad: límites de operación usados para detectar desviaciones.",
    "PoF": "Probabilidad de falla.",
    "CoF": "Consecuencia de falla, representada en la escala usada por el modelo.",
    "DRBI": "Inspección Basada en Riesgo Dinámica. Riesgo = PoF posterior × CoF.",
    "MDP": "Proceso de Decisión de Markov usado para seleccionar la acción prescriptiva.",
    "RVO": "Ventana/Restricción de oportunidad para definir plazo y factibilidad de intervención.",
    "MILP": "Programación Lineal Entera Mixta usada para asignar semana y recursos.",
    "Horizonte rodante": "Revisión periódica del programa con nueva evidencia y riesgo actualizado.",
}

# =============================================================================
# 1. PARÁMETROS MATEMÁTICOS DEL MODELO BASE
# =============================================================================
# Los valores de esta sección reproducen los parámetros identificados en el
# código base. No se modifican en la reorganización de arquitectura.

MAPA_IOW = {
    "pH": "PH",
    "Cloruros_ppm": "CL",
    "Salinidad_PTB": "SAL",
    "TAN_mgKOH_g": "TAN",
    "Hierro_mg_L": "FE",
    "BSW_pct": "BSW",
    "Temperatura_C": "TEMP",
    "Presion_bar_g": "PRES",
    "Velocidad_corrosion_mpy": "CR",
    "Inhibidor_ppm": "INH",
}

IOW_NOMBRES = {
    "PH": "pH",
    "CL": "Cloruros (ppm)",
    "SAL": "Salinidad (PTB)",
    "TAN": "Número ácido total - TAN (mg KOH/g)",
    "FE": "Hierro disuelto (mg/L)",
    "BSW": "Agua y sedimentos - BSW (%)",
    "TEMP": "Temperatura (°C)",
    "PRES": "Presión (bar g)",
    "CR": "Velocidad de corrosión (mpy)",
    "INH": "Dosificación de inhibidor (ppm)",
}

IOW_DEFAULT: Dict[str, Tuple[float, float]] = {
    "PH": (5.5, 7.5),
    "CL": (0.0, 30.0),
    "SAL": (0.0, 10.0),
    "TAN": (0.0, 1.0),
    "FE": (0.0, 3.0),
    "BSW": (0.0, 1.0),
    "TEMP": (20.0, 180.0),
    "PRES": (0.0, 25.0),
    "CR": (0.0, 3.0),
    "INH": (5.0, 80.0),
}

# Evidencia IOW = 0.45 Magnitud + 0.30 Duración + 0.25 Frecuencia
PESO_MAGNITUD = 0.45
PESO_DURACION = 0.30
PESO_FRECUENCIA = 0.25
VENTANA_FRECUENCIA_SEMANAS = 8
DURACION_SATURACION_SEMANAS = 4

POF_PRIOR_DEFAULT = {
    "T-01": 0.10,
    "T-02": 0.05,
    "T-03": 0.20,
    "T-04": 0.08,
    "T-05": 0.04,
    "T-06": 0.15,
    "T-07": 0.03,
    "T-08": 0.10,
}

COF_DEFAULT = {
    "T-01": 5.0,
    "T-02": 3.0,
    "T-03": 5.0,
    "T-04": 4.0,
    "T-05": 3.0,
    "T-06": 4.0,
    "T-07": 3.0,
    "T-08": 4.0,
}

TIPO_DEFAULT = {
    "T-01": "Submarina",
    "T-02": "Submarina",
    "T-03": "Submarina",
    "T-04": "Transferencia",
    "T-05": "Transferencia",
    "T-06": "Proceso",
    "T-07": "Proceso",
    "T-08": "Proceso",
}

# Factores y costos de demostración del código base.
FACTOR_SERVICIO = {
    "T-01": 1.15,
    "T-02": 1.05,
    "T-03": 1.25,
    "T-04": 1.15,
    "T-05": 1.00,
    "T-06": 1.20,
    "T-07": 1.05,
    "T-08": 1.20,
}

COSTO_INDISPONIBILIDAD_H = {
    "T-01": 4200,
    "T-02": 3800,
    "T-03": 5000,
    "T-04": 3200,
    "T-05": 3500,
    "T-06": 2800,
    "T-07": 4000,
    "T-08": 3000,
}

# =============================================================================
# 2. CATÁLOGO DE ACCIONES DEL MDP
# =============================================================================

ACCIONES = {
    "A1": {
        "nombre": "Vigilancia de condición reforzada",
        "objetivo": (
            "Mantener el tramo en operación incrementando el seguimiento de "
            "IOWs, variables de corrosión y tendencias de integridad."
        ),
        "tipo_intervencion": "Monitoreo basado en condición",
        "especialidad": "Operaciones / Corrosión / Integridad",
        "requiere_parada": "NO",
        "requiere_aislamiento": "NO",
        "alcance": (
            "Incrementar la frecuencia de revisión de variables de proceso, "
            "corrosión, laboratorio e indicadores de degradación."
        ),
    },
    "A2": {
        "nombre": "Inspección de integridad dirigida",
        "objetivo": (
            "Reducir la incertidumbre sobre la condición real del tramo con "
            "técnicas de inspección apropiadas al mecanismo de daño."
        ),
        "tipo_intervencion": "Inspección basada en riesgo",
        "especialidad": "Inspección / Integridad / NDE",
        "requiere_parada": "DEPENDE",
        "requiere_aislamiento": "DEPENDE",
        "alcance": (
            "UT de espesores, mapeo de corrosión, perfilometría, PAUT, "
            "radiografía u otra técnica NDE aplicable."
        ),
    },
    "A3": {
        "nombre": "Mitigación operativa y control de corrosión",
        "objetivo": (
            "Reducir temporalmente la severidad del mecanismo de daño mientras "
            "se mantiene controlado el riesgo."
        ),
        "tipo_intervencion": "Mitigación operacional",
        "especialidad": "Operaciones / Procesos / Corrosión",
        "requiere_parada": "NO",
        "requiere_aislamiento": "NO",
        "alcance": (
            "Evaluar reducción de carga, ajuste de variables operativas o "
            "dosificación química relacionada con la corrosión."
        ),
    },
    "A4": {
        "nombre": "Mantenimiento basado en condición / predictivo",
        "objetivo": (
            "Programar una intervención usando la tendencia de degradación, "
            "velocidad de corrosión, espesor, PoF y vida remanente."
        ),
        "tipo_intervencion": "Mantenimiento basado en condición",
        "especialidad": "Mantenimiento / Integridad / Corrosión",
        "requiere_parada": "DEPENDE",
        "requiere_aislamiento": "DEPENDE",
        "alcance": (
            "Preparar una intervención anticipada según pronóstico de condición "
            "y evolución del riesgo."
        ),
    },
    "A5": {
        "nombre": "Mantenimiento preventivo planificado",
        "objetivo": (
            "Preservar o restablecer la condición del tramo mediante una "
            "intervención programada antes de alcanzar una situación crítica."
        ),
        "tipo_intervencion": "Mantenimiento preventivo",
        "especialidad": "Mantenimiento mecánico / Integridad",
        "requiere_parada": "PROBABLE",
        "requiere_aislamiento": "SI",
        "alcance": (
            "Limpieza, retiro de depósitos, sustitución localizada, renovación "
            "de accesorios o intervención preventiva definida por Integridad."
        ),
    },
    "A6": {
        "nombre": "Intervención correctiva prioritaria de integridad",
        "objetivo": (
            "Restablecer la integridad mecánica cuando el deterioro y el riesgo "
            "requieren una intervención física prioritaria."
        ),
        "tipo_intervencion": "Mantenimiento correctivo planificado/prioritario",
        "especialidad": "Mantenimiento mecánico / Soldadura / Integridad / NDE",
        "requiere_parada": "SI",
        "requiere_aislamiento": "SI",
        "alcance": (
            "Evaluación detallada, reparación, sustitución de spool/componente "
            "o renovación del tramo según ingeniería e inspección."
        ),
    },
}

ESTADOS = ["BAJO", "MEDIO", "ALTO", "CRITICO"]

# Matrices de transición P(s'|s,a), idénticas a las del código base.
P_TRANS = {
    "A1": np.array(
        [[0.90, 0.09, 0.01, 0.00], [0.05, 0.72, 0.18, 0.05], [0.00, 0.05, 0.65, 0.30], [0.00, 0.00, 0.08, 0.92]]
    ),
    "A2": np.array(
        [[0.94, 0.055, 0.005, 0.00], [0.22, 0.68, 0.09, 0.01], [0.05, 0.30, 0.55, 0.10], [0.01, 0.10, 0.32, 0.57]]
    ),
    "A3": np.array(
        [[0.95, 0.045, 0.005, 0.00], [0.25, 0.68, 0.06, 0.01], [0.08, 0.42, 0.45, 0.05], [0.02, 0.18, 0.50, 0.30]]
    ),
    "A4": np.array(
        [[0.97, 0.03, 0.00, 0.00], [0.45, 0.50, 0.05, 0.00], [0.20, 0.55, 0.23, 0.02], [0.05, 0.30, 0.50, 0.15]]
    ),
    "A5": np.array(
        [[0.985, 0.015, 0.000, 0.000], [0.65, 0.33, 0.02, 0.00], [0.45, 0.45, 0.09, 0.01], [0.15, 0.45, 0.35, 0.05]]
    ),
    "A6": np.array(
        [[0.995, 0.005, 0.000, 0.000], [0.82, 0.18, 0.00, 0.00], [0.70, 0.27, 0.03, 0.00], [0.50, 0.40, 0.09, 0.01]]
    ),
}

COSTO_RIESGO = np.array([1.0, 5.0, 15.0, 40.0])
COSTO_ACCION = {"A1": 0.4, "A2": 2.0, "A3": 3.5, "A4": 5.0, "A5": 10.0, "A6": 18.0}
GAMMA_MDP = 0.90
TOLERANCIA_MDP = 1e-8
MAX_ITER_MDP = 10000

PRIORIDAD = {
    "BAJO": "P4 - RUTINARIA",
    "MEDIO": "P3 - PROGRAMABLE",
    "ALTO": "P2 - PRIORITARIA",
    "CRITICO": "P1 - URGENTE",
}

URGENCIA = {
    "BAJO": "U4 - SEGUIMIENTO",
    "MEDIO": "U3 - PLANIFICABLE",
    "ALTO": "U2 - PRIORITARIA",
    "CRITICO": "U1 - INMEDIATA / URGENTE",
}

PLAZO_RVO_DIAS = {"BAJO": 180, "MEDIO": 90, "ALTO": 30, "CRITICO": 7}
LEAD_TIME_DIAS = {"A1": 0, "A2": 7, "A3": 1, "A4": 14, "A5": 21, "A6": 7}
JERARQUIA_RIESGO = {"BAJO": 1, "MEDIO": 2, "ALTO": 3, "CRITICO": 4}
JERARQUIA_ACCION = {"A1": 1, "A2": 2, "A3": 3, "A4": 4, "A5": 5, "A6": 6}

# =============================================================================
# 3. RECURSOS, COSTEO Y RESTRICCIONES DE PROGRAMACIÓN
# =============================================================================

BASE_RECURSOS = {
    "A1": {"duracion_h": 2, "hh_mecanico": 0, "hh_soldador": 0, "hh_inspector": 2, "hh_corrosion": 4, "hh_planificador": 1, "hh_operaciones": 4},
    "A2": {"duracion_h": 8, "hh_mecanico": 8, "hh_soldador": 0, "hh_inspector": 16, "hh_corrosion": 4, "hh_planificador": 4, "hh_operaciones": 4},
    "A3": {"duracion_h": 6, "hh_mecanico": 4, "hh_soldador": 0, "hh_inspector": 2, "hh_corrosion": 8, "hh_planificador": 3, "hh_operaciones": 12},
    "A4": {"duracion_h": 16, "hh_mecanico": 32, "hh_soldador": 16, "hh_inspector": 12, "hh_corrosion": 8, "hh_planificador": 8, "hh_operaciones": 8},
    "A5": {"duracion_h": 32, "hh_mecanico": 56, "hh_soldador": 32, "hh_inspector": 16, "hh_corrosion": 8, "hh_planificador": 12, "hh_operaciones": 12},
    "A6": {"duracion_h": 48, "hh_mecanico": 80, "hh_soldador": 48, "hh_inspector": 24, "hh_corrosion": 12, "hh_planificador": 16, "hh_operaciones": 16},
}

TARIFA_HH = {
    "HH_MECANICO": 70,
    "HH_SOLDADOR": 85,
    "HH_INSPECTOR": 120,
    "HH_CORROSION": 130,
    "HH_PLANIFICADOR": 75,
    "HH_OPERACIONES": 65,
}

MATERIAL_BASE = {"A1": 200, "A2": 800, "A3": 3500, "A4": 10000, "A5": 24000, "A6": 48000}
EQUIPO_BASE = {"A1": 300, "A2": 2500, "A3": 1200, "A4": 4500, "A5": 8500, "A6": 12000}
SERVICIO_BASE = {"A1": 0, "A2": 3500, "A3": 2500, "A4": 7000, "A5": 12000, "A6": 20000}
COSTO_HSE_BASE = {"A1": 100, "A2": 600, "A3": 500, "A4": 1200, "A5": 2500, "A6": 3500}
FACTOR_INDISPONIBILIDAD = {"NO": 0.0, "DEPENDE": 0.35, "PROBABLE": 0.75, "SI": 1.0}
USO_UT = {"A1": 0, "A2": 8, "A3": 0, "A4": 8, "A5": 8, "A6": 12}
USO_SOLDADURA = {"A1": 0, "A2": 0, "A3": 0, "A4": 8, "A5": 24, "A6": 36}

CAPACIDAD_SEMANAL = {
    "HH_MECANICO": 160,
    "HH_SOLDADOR": 80,
    "HH_INSPECTOR": 40,
    "HH_CORROSION": 24,
    "HH_PLANIFICADOR": 32,
    "HH_OPERACIONES": 80,
}
CAP_UT_H_SEMANA = 24
CAP_SOLDADURA_H_SEMANA = 40
MAX_FRENTES_SEMANA = 2
MAX_PARADAS_SIMULTANEAS = 1
PRESUPUESTO_MENSUAL_DEFAULT = 180000.0
HORIZONTE_SEMANAS_DEFAULT = 52
VENTANA_CONGELADA_SEMANAS_DEFAULT = 4
UMBRAL_INCREMENTO_RIESGO_DEFAULT = 0.10

# =============================================================================
# 4. ESTRUCTURA DE DATOS DE ENTRADA
# =============================================================================

REQUIRED = [
    "Fecha_hora",
    "Periodo_semana",
    "ID_tramo",
    "TIPO_TUBERIA",
    "Servicio_fluido",
    "Espesor_actual_mm",
    "Velocidad_corrosion_mpy",
    "Vida_remanente_anios",
    "pH",
    "Cloruros_ppm",
    "Salinidad_PTB",
    "TAN_mgKOH_g",
    "Hierro_mg_L",
    "BSW_pct",
    "Temperatura_C",
    "Presion_bar_g",
    "Inhibidor_ppm",
    "PoF_prior",
    "CoF_valor",
    "Diametro_in",
    "Longitud_m",
]

NUMERIC_REQUIRED = [c for c in REQUIRED if c not in {"Fecha_hora", "ID_tramo", "TIPO_TUBERIA", "Servicio_fluido"}]

ALIASES_COLUMNAS = {
    "fecha": "Fecha_hora",
    "fecha_hora": "Fecha_hora",
    "fecha/hora": "Fecha_hora",
    "periodo": "Periodo_semana",
    "periodo_semana": "Periodo_semana",
    "semana": "Periodo_semana",
    "id tramo": "ID_tramo",
    "id_tramo": "ID_tramo",
    "tramo": "ID_tramo",
    "tipo_tuberia": "TIPO_TUBERIA",
    "tipo tuberia": "TIPO_TUBERIA",
    "tipo tubería": "TIPO_TUBERIA",
    "servicio": "Servicio_fluido",
    "servicio_fluido": "Servicio_fluido",
    "espesor actual": "Espesor_actual_mm",
    "espesor_actual_mm": "Espesor_actual_mm",
    "velocidad corrosion mpy": "Velocidad_corrosion_mpy",
    "velocidad_corrosion_mpy": "Velocidad_corrosion_mpy",
    "vida remanente anios": "Vida_remanente_anios",
    "vida_remanente_anios": "Vida_remanente_anios",
    "cof": "CoF_valor",
    "cof_valor": "CoF_valor",
    "pof prior": "PoF_prior",
    "pof_prior": "PoF_prior",
    "diametro": "Diametro_in",
    "diámetro": "Diametro_in",
    "diametro_in": "Diametro_in",
    "longitud": "Longitud_m",
    "longitud_m": "Longitud_m",
}


@dataclass
class AuditRecord:
    etapa: str
    nivel: str
    mensaje: str
    detalle: str = ""


@dataclass
class DataQualityReport:
    registros_entrada: int = 0
    registros_salida: int = 0
    tramos: int = 0
    duplicados_eliminados: int = 0
    valores_interpolados: int = 0
    completitud_pct: float = 0.0
    advertencias: List[str] = field(default_factory=list)
    auditoria: List[AuditRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "registros_entrada": self.registros_entrada,
            "registros": self.registros_salida,
            "tramos": self.tramos,
            "duplicados_eliminados": self.duplicados_eliminados,
            "valores_interpolados": self.valores_interpolados,
            "calidad_pct": round(self.completitud_pct, 1),
            "advertencias": list(self.advertencias),
            "auditoria": [r.__dict__ for r in self.auditoria],
        }


# =============================================================================
# 5. DATOS DE DEMOSTRACIÓN Y PLANTILLAS
# =============================================================================

def demo_dataset(seed: int = 42, semanas: int = 16) -> pd.DataFrame:
    """Genera datos reproducibles para validar las tres familias de tubería."""
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    meta = {
        "T-01": ("Submarina", "Crudo", 12, 850, 8.5),
        "T-02": ("Submarina", "Nafta", 8, 640, 7.9),
        "T-03": ("Submarina", "Agua ácida / hidrocarburo", 10, 520, 7.6),
        "T-04": ("Transferencia", "Residual / Fuel Oil", 10, 900, 8.1),
        "T-05": ("Transferencia", "Diésel", 8, 760, 8.3),
        "T-06": ("Proceso", "Slop", 8, 180, 7.4),
        "T-07": ("Proceso", "Gasolina", 6, 220, 7.8),
        "T-08": ("Proceso", "Asfalto caliente", 10, 260, 8.0),
    }
    start = pd.Timestamp("2026-01-05")

    for tid, (tipo, servicio, diametro, longitud, espesor_inicial) in meta.items():
        severity = {"T-03": 1.55, "T-06": 1.35, "T-08": 1.25, "T-01": 1.15}.get(tid, 0.85)
        for semana in range(1, semanas + 1):
            trend = (semana - 1) / max(semanas - 1, 1)
            cr = max(0.3, rng.normal(1.4 * severity + 2.2 * trend * severity, 0.35))
            ph = rng.normal(6.5 - 0.9 * trend * severity, 0.18)
            cl = max(1, rng.normal(18 + 22 * trend * severity, 5))
            sal = max(0, rng.normal(4 + 4 * trend * severity, 1.3))
            tan = max(0, rng.normal(0.45 + 0.45 * trend * severity, 0.12))
            fe = max(0, rng.normal(1.1 + 2.4 * trend * severity, 0.45))
            bsw = max(0, rng.normal(0.35 + 0.6 * trend * severity, 0.15))
            temp = rng.normal(110 + 22 * trend * (1 if tipo == "Proceso" else 0.45), 6)
            pres = max(0.5, rng.normal(10 + 6 * trend, 1.1))
            inh = max(0, rng.normal(38 - 12 * trend * severity, 5))
            espesor = max(3.5, espesor_inicial - cr * 0.0254 * (semana / 52))
            vida = max(0.15, (espesor - 4.5) / (cr * 0.0254))

            rows.append(
                {
                    "Fecha_hora": start + pd.Timedelta(days=7 * (semana - 1)),
                    "Periodo_semana": semana,
                    "ID_tramo": tid,
                    "TIPO_TUBERIA": tipo,
                    "Servicio_fluido": servicio,
                    "Espesor_actual_mm": round(espesor, 3),
                    "Velocidad_corrosion_mpy": round(cr, 3),
                    "Vida_remanente_anios": round(vida, 2),
                    "pH": round(ph, 3),
                    "Cloruros_ppm": round(cl, 2),
                    "Salinidad_PTB": round(sal, 2),
                    "TAN_mgKOH_g": round(tan, 3),
                    "Hierro_mg_L": round(fe, 3),
                    "BSW_pct": round(bsw, 3),
                    "Temperatura_C": round(temp, 2),
                    "Presion_bar_g": round(pres, 2),
                    "Inhibidor_ppm": round(inh, 2),
                    "PoF_prior": POF_PRIOR_DEFAULT[tid],
                    "CoF_valor": COF_DEFAULT[tid],
                    "Diametro_in": diametro,
                    "Longitud_m": longitud,
                }
            )

    return pd.DataFrame(rows)


def demo_windows(tipo: str, dataset: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Crea ventanas editables para la demostración, una por tramo."""
    if tipo not in TIPOS_TUBERIA:
        raise ValueError(f"Tipo de tubería no válido: {tipo}")
    data = demo_dataset() if dataset is None else normalize_input(dataset)
    data = data[data["TIPO_TUBERIA"].eq(tipo)].copy()
    if data.empty:
        return pd.DataFrame(columns=["ID_tramo", "ID_ventana", "Inicio", "Fin", "Tipo", "Restricciones"])
    base_date = pd.to_datetime(data["Fecha_hora"].min()).normalize()
    out = []
    for j, tid in enumerate(sorted(data["ID_tramo"].dropna().astype(str).unique()), 1):
        inicio = base_date + pd.Timedelta(days=7 * (12 + j))
        out.append(
            {
                "ID_tramo": tid,
                "ID_ventana": f"W-{tid}-01",
                "Inicio": inicio,
                "Fin": inicio + pd.Timedelta(hours=48),
                "Tipo": "Ventana de mantenimiento propuesta",
                "Restricciones": "Validar con Operaciones, Integridad, permisos y disponibilidad de recursos.",
            }
        )
    return pd.DataFrame(out)


# =============================================================================
# 6. LECTURA DE ARCHIVOS XML / EXCEL / CSV
# =============================================================================

def parse_xml(path: str) -> pd.DataFrame:
    """Lee XML tabular tolerando varias estructuras comunes."""
    path = str(path)
    errores = []
    for xpath in ["//registro", "//row", "//Record", "//item", "//data/*"]:
        try:
            df = pd.read_xml(path, xpath=xpath)
            if df is not None and len(df) > 0 and len(df.columns) > 2:
                return df
        except Exception as exc:
            errores.append(f"{xpath}: {exc}")

    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    records: List[dict] = []
    for node in root.iter():
        children = list(node)
        if children and all(len(list(c)) == 0 for c in children):
            rec = {c.tag.split("}")[-1]: c.text for c in children}
            if len(rec) >= 3:
                records.append(rec)
    if not records:
        raise ValueError(
            "No se identificaron registros tabulares en el XML. "
            "Use la plantilla incluida en el DSS o revise la estructura del archivo."
        )
    return pd.DataFrame(records)


def load_file(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Carga XML, XLSX/XLS o CSV. Para Excel, usa la hoja indicada o la primera."""
    if not path:
        raise ValueError("No se recibió una ruta de archivo.")
    ext = Path(path).suffix.lower()
    if ext == ".xml":
        return parse_xml(path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    if ext == ".csv":
        # Intenta UTF-8 y, si falla, latin-1.
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    raise ValueError("Formato no soportado. Use XML, XLSX, XLS o CSV.")


# =============================================================================
# 7. NORMALIZACIÓN Y VALIDACIÓN DE DATOS
# =============================================================================

def _clean_column_name(name: object) -> str:
    txt = str(name).strip()
    key = re.sub(r"\s+", " ", txt).lower()
    return ALIASES_COLUMNAS.get(key, txt)


def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres, tipos y campos inferibles sin alterar valores válidos."""
    if df is None:
        raise ValueError("La tabla de entrada está vacía.")
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    x = df.copy()
    x.columns = [_clean_column_name(c) for c in x.columns]

    if "ID_tramo" in x.columns:
        x["ID_tramo"] = x["ID_tramo"].astype(str).str.strip()

    # Inferencias permitidas por el prototipo de demostración.
    if "TIPO_TUBERIA" not in x.columns and "ID_tramo" in x.columns:
        x["TIPO_TUBERIA"] = x["ID_tramo"].map(TIPO_DEFAULT)
    if "PoF_prior" not in x.columns and "ID_tramo" in x.columns:
        x["PoF_prior"] = x["ID_tramo"].map(POF_PRIOR_DEFAULT)
    if "CoF_valor" not in x.columns and "ID_tramo" in x.columns:
        x["CoF_valor"] = x["ID_tramo"].map(COF_DEFAULT)

    if "TIPO_TUBERIA" in x.columns:
        # Estandariza capitalización y elimina espacios.
        mapa_tipo = {t.lower(): t for t in TIPOS_TUBERIA}
        tipo_original = x["TIPO_TUBERIA"].astype(str).str.strip()
        tipo_mapeado = tipo_original.str.lower().map(mapa_tipo)
        x["TIPO_TUBERIA"] = tipo_mapeado.where(tipo_mapeado.notna(), tipo_original)

    if "Fecha_hora" in x.columns:
        x["Fecha_hora"] = pd.to_datetime(x["Fecha_hora"], errors="coerce")

    if "Periodo_semana" not in x.columns and {"Fecha_hora", "ID_tramo"}.issubset(x.columns):
        ordered = x.sort_values(["ID_tramo", "Fecha_hora"]).copy()
        ordered["Periodo_semana"] = ordered.groupby("ID_tramo").cumcount() + 1
        x["Periodo_semana"] = ordered["Periodo_semana"].reindex(x.index)

    for col in NUMERIC_REQUIRED:
        if col in x.columns:
            x[col] = pd.to_numeric(x[col], errors="coerce")

    return x


def validate_input(df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityReport]:
    """Valida integridad estructural y consistencia básica del dataset."""
    x = normalize_input(df)
    report = DataQualityReport(registros_entrada=len(x))

    missing = [c for c in REQUIRED if c not in x.columns]
    if missing:
        raise ValueError(
            "Faltan columnas obligatorias: " + ", ".join(missing) + ". "
            "Puede usar la plantilla manual o XML incluida para conservar los nombres esperados."
        )

    if x.empty:
        raise ValueError("El dataset no contiene registros.")

    if x["ID_tramo"].isna().any() or x["ID_tramo"].astype(str).str.strip().eq("").any():
        raise ValueError("Existen registros sin ID_tramo.")

    if x["Fecha_hora"].isna().any():
        n = int(x["Fecha_hora"].isna().sum())
        raise ValueError(f"Existen {n} registros con Fecha_hora no interpretable.")

    tipos_invalidos = sorted(set(x.loc[~x["TIPO_TUBERIA"].isin(TIPOS_TUBERIA), "TIPO_TUBERIA"].astype(str)))
    if tipos_invalidos:
        raise ValueError(
            "Se encontraron tipos de tubería no válidos: " + ", ".join(tipos_invalidos) +
            ". Valores permitidos: " + ", ".join(TIPOS_TUBERIA)
        )

    if x["PoF_prior"].isna().any():
        ids = sorted(x.loc[x["PoF_prior"].isna(), "ID_tramo"].unique().tolist())
        raise ValueError("Hay tramos sin PoF_prior: " + ", ".join(map(str, ids)))
    if ((x["PoF_prior"] < 0) | (x["PoF_prior"] > 1)).any():
        raise ValueError("PoF_prior debe estar entre 0 y 1.")

    if x["CoF_valor"].isna().any():
        ids = sorted(x.loc[x["CoF_valor"].isna(), "ID_tramo"].unique().tolist())
        raise ValueError("Hay tramos sin CoF_valor: " + ", ".join(map(str, ids)))
    if (x["CoF_valor"] <= 0).any():
        raise ValueError("CoF_valor debe ser mayor que cero.")

    # Verifica consistencia de tipo por ID.
    tipo_count = x.groupby("ID_tramo")["TIPO_TUBERIA"].nunique(dropna=False)
    inconsistentes = tipo_count[tipo_count > 1].index.tolist()
    if inconsistentes:
        raise ValueError(
            "Un mismo ID_tramo aparece asociado a más de un tipo de tubería: " + ", ".join(map(str, inconsistentes))
        )

    # Advertencias no bloqueantes.
    for col in ["Espesor_actual_mm", "Velocidad_corrosion_mpy", "Vida_remanente_anios", "Diametro_in", "Longitud_m"]:
        if (x[col] < 0).any():
            report.advertencias.append(f"{col}: se detectaron valores negativos que deben revisarse.")

    if (x["CoF_valor"] > 5).any():
        report.advertencias.append("Existen valores CoF > 5; verifique que la escala usada sea compatible con el modelo.")

    report.tramos = int(x["ID_tramo"].nunique())
    report.completitud_pct = float(100 * (1 - x[REQUIRED].isna().mean().mean()))
    report.auditoria.append(AuditRecord("Entrada", "OK", "Estructura de columnas validada."))
    return x, report


# =============================================================================
# 8. CAPA 1 - ADQUISICIÓN / IIoT / ETL
# =============================================================================

def layer1_etl(df: pd.DataFrame, tipo: str) -> Tuple[pd.DataFrame, dict]:
    """Selecciona el tipo de tubería y depura la información antes del cálculo."""
    if tipo not in TIPOS_TUBERIA:
        raise ValueError(f"Tipo de tubería no válido: {tipo}")

    x, quality = validate_input(df)
    x = x[x["TIPO_TUBERIA"].eq(tipo)].copy()
    if x.empty:
        raise ValueError(f"El archivo no contiene registros para tuberías de tipo {tipo}.")

    # Orden temporal y eliminación controlada de duplicados exactos de tramo-fecha.
    before = len(x)
    x = (
        x.sort_values(["ID_tramo", "Fecha_hora", "Periodo_semana"])
        .drop_duplicates(subset=["ID_tramo", "Fecha_hora"], keep="last")
        .reset_index(drop=True)
    )
    quality.duplicados_eliminados = before - len(x)

    # Las variables IOW pueden ser interpoladas dentro del mismo tramo. No se
    # interpolan PoF, CoF, geometría ni datos de identificación.
    variables_interp = list(MAPA_IOW.keys())
    valores_interpolados = 0
    for col in variables_interp:
        faltantes_antes = int(x[col].isna().sum())
        if faltantes_antes:
            x[col] = x.groupby("ID_tramo", group_keys=False)[col].apply(
                lambda s: s.interpolate(limit_direction="both")
            )
            faltantes_despues = int(x[col].isna().sum())
            valores_interpolados += max(0, faltantes_antes - faltantes_despues)
    quality.valores_interpolados = valores_interpolados

    # Tras la interpolación, una variable IOW todavía vacía bloquea el cálculo.
    faltantes_iow = {c: int(x[c].isna().sum()) for c in MAPA_IOW if x[c].isna().any()}
    if faltantes_iow:
        detalle = ", ".join(f"{k}={v}" for k, v in faltantes_iow.items())
        raise ValueError(
            "Persisten valores faltantes en variables IOW después de la depuración: " + detalle
        )

    quality.registros_salida = len(x)
    quality.tramos = int(x["ID_tramo"].nunique())
    quality.completitud_pct = float(100 * (1 - x[REQUIRED].isna().mean().mean()))
    quality.auditoria.append(
        AuditRecord(
            "Capa 1",
            "OK",
            f"Datos filtrados para {tipo} y ordenados por tramo/fecha.",
            f"Duplicados eliminados={quality.duplicados_eliminados}; valores IOW interpolados={quality.valores_interpolados}.",
        )
    )

    return x, quality.to_dict()


# =============================================================================
# 9. CAPA 2 - IOWs: DESVIACIONES, MAGNITUD, DURACIÓN Y FRECUENCIA
# =============================================================================

def _magnitud_desviacion(valor: float, limite_inferior: float, limite_superior: float) -> float:
    """Fórmula de magnitud normalizada usada en el código base."""
    if pd.isna(valor):
        return 0.0
    rango = limite_superior - limite_inferior
    if rango <= 0:
        rango = max(abs(limite_superior), 1)
    if valor < limite_inferior:
        magnitud = (limite_inferior - valor) / rango
    elif valor > limite_superior:
        magnitud = (valor - limite_superior) / rango
    else:
        magnitud = 0.0
    return min(max(float(magnitud), 0.0), 1.0)


def _racha_consecutiva(serie: pd.Series) -> pd.Series:
    contador = 0
    resultado: List[int] = []
    for valor in serie:
        contador = contador + 1 if valor == 1 else 0
        resultado.append(contador)
    return pd.Series(resultado, index=serie.index)


def validate_iow_params(iow_params: Mapping[str, Sequence[float]]) -> Dict[str, Tuple[float, float]]:
    p: Dict[str, Tuple[float, float]] = {}
    faltantes = [cod for cod in MAPA_IOW.values() if cod not in iow_params]
    if faltantes:
        raise ValueError("Faltan límites IOW para: " + ", ".join(faltantes))
    for cod in MAPA_IOW.values():
        vals = iow_params[cod]
        if len(vals) != 2:
            raise ValueError(f"IOW {cod}: se requieren límite inferior y superior.")
        li, ls = float(vals[0]), float(vals[1])
        if not np.isfinite(li) or not np.isfinite(ls):
            raise ValueError(f"IOW {cod}: límites no numéricos.")
        if li > ls:
            raise ValueError(f"IOW {cod}: el límite inferior no puede superar al límite superior.")
        p[cod] = (li, ls)
    return p


def layer2_iows(df: pd.DataFrame, iow_params: Optional[Mapping[str, Sequence[float]]] = None) -> pd.DataFrame:
    p = validate_iow_params(iow_params or IOW_DEFAULT)
    x = df.copy().sort_values(["ID_tramo", "Periodo_semana"]).reset_index(drop=True)

    columnas_desviacion: List[str] = []
    columnas_magnitud: List[str] = []
    columnas_duracion: List[str] = []
    columnas_frecuencia: List[str] = []

    for columna_dataset, codigo_iow in MAPA_IOW.items():
        li, ls = p[codigo_iow]
        col_dev = f"DEV_{codigo_iow}"
        col_mag = f"MAG_{codigo_iow}"
        col_dur = f"DUR_{codigo_iow}"
        col_freq = f"FREQ_{codigo_iow}"

        x[col_dev] = ((x[columna_dataset] < li) | (x[columna_dataset] > ls)).astype(int)
        x[col_mag] = x[columna_dataset].apply(lambda v: _magnitud_desviacion(v, li, ls))
        x[col_dur] = x.groupby("ID_tramo")[col_dev].transform(_racha_consecutiva)
        x[col_dur] = (x[col_dur] / DURACION_SATURACION_SEMANAS).clip(0, 1)
        x[col_freq] = x.groupby("ID_tramo")[col_dev].transform(
            lambda s: s.rolling(window=VENTANA_FRECUENCIA_SEMANAS, min_periods=1).mean()
        )

        columnas_desviacion.append(col_dev)
        columnas_magnitud.append(col_mag)
        columnas_duracion.append(col_dur)
        columnas_frecuencia.append(col_freq)

    x["N_IOW_DESVIADAS"] = x[columnas_desviacion].sum(axis=1)
    x["INDICE_IOW"] = x["N_IOW_DESVIADAS"] / len(columnas_desviacion)
    x["NIVEL_IOW"] = x["N_IOW_DESVIADAS"].apply(
        lambda n: "NORMAL" if n == 0 else ("ALERTA" if n <= 2 else ("ALTA" if n <= 4 else "CRITICA"))
    )

    x["M_t"] = x[columnas_magnitud].mean(axis=1)
    x["D_t"] = x[columnas_duracion].mean(axis=1)
    x["F_t"] = x[columnas_frecuencia].mean(axis=1)

    # Fórmula exacta del modelo base.
    x["E_t"] = (
        PESO_MAGNITUD * x["M_t"]
        + PESO_DURACION * x["D_t"]
        + PESO_FRECUENCIA * x["F_t"]
    ).clip(0, 1)

    x["NIVEL_EVIDENCIA"] = x["E_t"].apply(
        lambda e: "SIN EVIDENCIA" if e == 0 else ("BAJA" if e <= 0.25 else ("MEDIA" if e <= 0.50 else ("ALTA" if e <= 0.75 else "CRITICA")))
    )
    return x


# =============================================================================
# 10. CAPA 3 - ACTUALIZACIÓN BAYESIANA DE PoF
# =============================================================================

def layer3_bayes(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["PoF_prior"] = pd.to_numeric(x["PoF_prior"], errors="coerce")
    if x["PoF_prior"].isna().any():
        raise ValueError("No es posible ejecutar Bayes porque existen PoF_prior no numéricas.")

    # Verosimilitudes del código base:
    # E=0 -> 0.50 / 0.50; E=1 -> 0.80 / 0.20.
    x["P_E_dado_F"] = 0.50 + 0.30 * x["E_t"]
    x["P_E_dado_noF"] = 0.50 - 0.30 * x["E_t"]

    numerador = x["P_E_dado_F"] * x["PoF_prior"]
    denominador = numerador + x["P_E_dado_noF"] * (1 - x["PoF_prior"])
    if (denominador <= 0).any():
        raise ValueError("Se obtuvo un denominador bayesiano no positivo. Revise PoF_prior y E_t.")

    x["PoF_posterior"] = (numerador / denominador).clip(0, 1).round(4)
    x["Delta_PoF"] = (x["PoF_posterior"] - x["PoF_prior"]).round(4)
    x["NIVEL_PoF"] = x["PoF_posterior"].apply(
        lambda p: "MUY BAJA" if p < 0.05 else ("BAJA" if p < 0.10 else ("MEDIA" if p < 0.20 else ("ALTA" if p < 0.35 else "MUY ALTA")))
    )
    return x


# =============================================================================
# 11. CAPA 4 - DRBI: RIESGO DINÁMICO
# =============================================================================

def layer4_drbi(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["CoF_valor"] = pd.to_numeric(x["CoF_valor"], errors="coerce")
    if x["CoF_valor"].isna().any():
        raise ValueError("No es posible calcular DRBI porque existen CoF_valor no numéricas.")

    x["Riesgo_DRBI"] = (x["PoF_posterior"] * x["CoF_valor"]).round(4)
    x["Riesgo_normalizado"] = (x["Riesgo_DRBI"] / 5.0).clip(0, 1).round(4)
    x["NIVEL_RIESGO"] = x["Riesgo_DRBI"].apply(
        lambda r: "BAJO" if r < 0.25 else ("MEDIO" if r < 0.60 else ("ALTO" if r < 1.00 else "CRITICO"))
    )
    return x


# =============================================================================
# 12. CAPA 5A - MDP: POLÍTICA PRESCRIPTIVA
# =============================================================================

def validate_mdp_matrices() -> None:
    for action_id, matrix in P_TRANS.items():
        if matrix.shape != (len(ESTADOS), len(ESTADOS)):
            raise ValueError(f"Matriz {action_id}: forma inválida {matrix.shape}.")
        if not np.allclose(matrix.sum(axis=1), np.ones(len(ESTADOS)), atol=1e-10):
            raise ValueError(f"Matriz {action_id}: cada fila debe sumar 1.")
        if (matrix < 0).any() or (matrix > 1).any():
            raise ValueError(f"Matriz {action_id}: contiene probabilidades fuera de [0,1].")


def mdp_policy() -> Tuple[dict, pd.DataFrame]:
    """Value iteration idéntica a la del código base."""
    validate_mdp_matrices()
    action_ids = list(ACCIONES.keys())

    # C(s,a) = costo de acción + riesgo esperado posterior a la acción.
    C = np.zeros((len(ESTADOS), len(action_ids)))
    for s in range(len(ESTADOS)):
        for a, action_id in enumerate(action_ids):
            riesgo_residual_esperado = np.dot(P_TRANS[action_id][s], COSTO_RIESGO)
            C[s, a] = COSTO_ACCION[action_id] + riesgo_residual_esperado

    V = np.zeros(len(ESTADOS))
    convergio_en = MAX_ITER_MDP
    for iteracion in range(MAX_ITER_MDP):
        Q = np.zeros_like(C)
        for s in range(len(ESTADOS)):
            for a, action_id in enumerate(action_ids):
                Q[s, a] = C[s, a] + GAMMA_MDP * np.dot(P_TRANS[action_id][s], V)
        V_nuevo = Q.min(axis=1)
        diferencia = np.max(np.abs(V_nuevo - V))
        V = V_nuevo.copy()
        if diferencia < TOLERANCIA_MDP:
            convergio_en = iteracion + 1
            break

    # Recalcular Q con V convergido.
    Q = np.zeros_like(C)
    for s in range(len(ESTADOS)):
        for a, action_id in enumerate(action_ids):
            Q[s, a] = C[s, a] + GAMMA_MDP * np.dot(P_TRANS[action_id][s], V)

    policy = {estado: action_ids[int(np.argmin(Q[s]))] for s, estado in enumerate(ESTADOS)}
    table = pd.DataFrame(
        {
            "Estado_DRBI": ESTADOS,
            "Accion_ID": [policy[e] for e in ESTADOS],
            "Estrategia_MDP": [ACCIONES[policy[e]]["nombre"] for e in ESTADOS],
            "Tipo_intervencion": [ACCIONES[policy[e]]["tipo_intervencion"] for e in ESTADOS],
            "Valor_estado": V.round(3),
            "Iteraciones_convergencia": convergio_en,
        }
    )
    return policy, table


# =============================================================================
# 13. CAPA 5B - RVO: PLAZO, CONSOLIDACIÓN Y VENTANAS
# =============================================================================

def _normalize_windows(windows: Optional[pd.DataFrame]) -> pd.DataFrame:
    cols = ["ID_tramo", "ID_ventana", "Inicio", "Fin", "Tipo", "Restricciones"]
    if windows is None or len(windows) == 0:
        return pd.DataFrame(columns=cols)
    w = windows.copy()
    ren = {
        "ID tramo": "ID_tramo",
        "ID ventana": "ID_ventana",
        "Fecha/hora inicio": "Inicio",
        "Fecha/hora fin": "Fin",
        "Tipo de ventana": "Tipo",
        "Restricciones": "Restricciones",
    }
    w = w.rename(columns={k: v for k, v in ren.items() if k in w.columns})
    missing = [c for c in cols if c not in w.columns]
    if missing:
        raise ValueError("La tabla de ventanas no contiene: " + ", ".join(missing))
    w["ID_tramo"] = w["ID_tramo"].astype(str).str.strip()
    w["Inicio"] = pd.to_datetime(w["Inicio"], errors="coerce")
    w["Fin"] = pd.to_datetime(w["Fin"], errors="coerce")
    if w[["Inicio", "Fin"]].isna().any().any():
        raise ValueError("Existen ventanas con fecha de inicio o fin no interpretable.")
    if (w["Fin"] < w["Inicio"]).any():
        raise ValueError("Existen ventanas cuyo fin es anterior al inicio.")
    return w[cols].copy()


def layer5_mdp_rvo(
    df: pd.DataFrame,
    windows: Optional[pd.DataFrame] = None,
    auto_generate_windows: bool = True,
    fecha_corte_analisis: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = df.copy()
    policy, policy_table = mdp_policy()
    x["ACCION_MDP_ID"] = x["NIVEL_RIESGO"].map(policy)

    for col, key in [
        ("ESTRATEGIA_MANTENIMIENTO", "nombre"),
        ("OBJETIVO_MANTENIMIENTO", "objetivo"),
        ("TIPO_INTERVENCION", "tipo_intervencion"),
        ("ALCANCE_TECNICO_PRELIMINAR", "alcance"),
        ("ESPECIALIDAD_REQUERIDA", "especialidad"),
        ("REQUIERE_PARADA", "requiere_parada"),
        ("REQUIERE_AISLAMIENTO", "requiere_aislamiento"),
    ]:
        x[col] = x["ACCION_MDP_ID"].map(lambda a: ACCIONES[a][key])

    x["PRIORIDAD_MTTO"] = x["NIVEL_RIESGO"].map(PRIORIDAD)
    x["URGENCIA_RVO"] = x["NIVEL_RIESGO"].map(URGENCIA)
    x["PLAZO_RVO_DIAS"] = x["NIVEL_RIESGO"].map(PLAZO_RVO_DIAS)
    x["LEAD_TIME_DIAS"] = x["ACCION_MDP_ID"].map(LEAD_TIME_DIAS)
    x["FECHA_DISPONIBLE_DESDE"] = x["Fecha_hora"] + pd.to_timedelta(x["LEAD_TIME_DIAS"], unit="D")
    x["FECHA_LIMITE_RVO"] = x["Fecha_hora"] + pd.to_timedelta(x["PLAZO_RVO_DIAS"], unit="D")
    x["DIAS_VIDA_REMANENTE"] = x["Vida_remanente_anios"] * 365.25
    x["ALERTA_VIDA_REMANENTE"] = np.where(
        x["DIAS_VIDA_REMANENTE"] <= x["PLAZO_RVO_DIAS"],
        "REVISIÓN INMEDIATA DE INTEGRIDAD",
        "SIN CONFLICTO APARENTE",
    )

    # Consolidación del código base: máximo nivel de riesgo por tramo, luego
    # mayor riesgo y acción más intensa en caso de empate.
    x["JERARQUIA_RIESGO"] = x["NIVEL_RIESGO"].map(JERARQUIA_RIESGO)
    x["JERARQUIA_ACCION"] = x["ACCION_MDP_ID"].map(JERARQUIA_ACCION)
    ordered = x.sort_values(
        ["ID_tramo", "JERARQUIA_RIESGO", "Riesgo_DRBI", "JERARQUIA_ACCION"],
        ascending=[True, False, False, False],
    )
    interventions = ordered.groupby("ID_tramo", as_index=False).first()

    # Primera fecha en que apareció el máximo nivel seleccionado.
    detection_dates = []
    for _, row in interventions.iterrows():
        records = x[(x["ID_tramo"] == row["ID_tramo"]) & (x["NIVEL_RIESGO"] == row["NIVEL_RIESGO"])]
        detection_dates.append(records["Fecha_hora"].min())
    interventions["FECHA_DETECCION"] = pd.to_datetime(detection_dates)

    # BASE TEMPORAL DE PROGRAMACIÓN (v2.3)
    # --------------------------------------
    # La primera detección se conserva como dato histórico de trazabilidad, pero
    # el programa que se genera HOY dentro del horizonte analizado no debe volver
    # a calendarizar actividades en fechas anteriores a la evidencia más reciente.
    # Por ello se define una FECHA DE CORTE DE ANÁLISIS: por defecto es la fecha
    # más reciente disponible en el conjunto seleccionado. Todas las fechas de
    # preparación y RVO del nuevo ciclo se derivan de esa fecha de corte.
    if fecha_corte_analisis is None:
        fecha_corte_analisis = pd.Timestamp(x["Fecha_hora"].max()).normalize()
    fecha_corte_analisis = pd.Timestamp(fecha_corte_analisis).normalize()
    interventions["FECHA_CORTE_ANALISIS"] = fecha_corte_analisis
    interventions["FECHA_ULTIMA_EVIDENCIA_TRAMO"] = interventions["ID_tramo"].map(
        x.groupby("ID_tramo")["Fecha_hora"].max()
    )
    interventions["ANTIGUEDAD_DATOS_DIAS"] = (
        interventions["FECHA_CORTE_ANALISIS"] - interventions["FECHA_ULTIMA_EVIDENCIA_TRAMO"].dt.normalize()
    ).dt.days.clip(lower=0)

    # Fecha más temprana = fecha de corte + lead time técnico de la acción MDP.
    # Fecha límite RVO = fecha de corte + plazo máximo seguro asociado al nivel
    # de riesgo. De esta manera las fechas son reproducibles y auditables.
    interventions["FECHA_DISPONIBLE_DESDE"] = interventions["FECHA_CORTE_ANALISIS"] + pd.to_timedelta(
        interventions["LEAD_TIME_DIAS"], unit="D"
    )
    interventions["FECHA_LIMITE_PLANIFICACION"] = interventions["FECHA_CORTE_ANALISIS"] + pd.to_timedelta(
        interventions["PLAZO_RVO_DIAS"], unit="D"
    )
    interventions["FECHA_LIMITE_RVO"] = interventions["FECHA_LIMITE_PLANIFICACION"]
    interventions["BASE_FECHA_PROGRAMACION"] = (
        "Fecha de corte (última evidencia del análisis) + lead time / plazo RVO"
    )
    interventions["INGRESA_MILP"] = np.where(interventions["ACCION_MDP_ID"].eq("A1"), "NO", "SI")
    interventions["ID_SOLICITUD"] = [f"INT-{i + 1:03d}" for i in range(len(interventions))]

    window_table = _normalize_windows(windows)
    if window_table.empty and auto_generate_windows:
        generated = []
        for _, row in interventions.iterrows():
            if row["ACCION_MDP_ID"] in {"A1", "A3"}:
                continue
            # Se crea una ventana de planificación en la primera fecha factible.
            start = row["FECHA_DISPONIBLE_DESDE"]
            generated.append(
                {
                    "ID_tramo": row["ID_tramo"],
                    "ID_ventana": f"W-{row['ID_tramo']}-AUTO",
                    "Inicio": start,
                    "Fin": start + pd.Timedelta(hours=max(24, BASE_RECURSOS[row["ACCION_MDP_ID"]]["duracion_h"])),
                    "Tipo": "Ventana provisional calculada por el DSS",
                    "Restricciones": "Fecha propuesta a partir de fecha de corte + lead time. Debe confirmarse con Operaciones, Integridad, HSE y Planificación antes de ejecutar.",
                }
            )
        window_table = pd.DataFrame(generated, columns=["ID_tramo", "ID_ventana", "Inicio", "Fin", "Tipo", "Restricciones"])

    results = []
    for _, row in interventions.iterrows():
        action = row["ACCION_MDP_ID"]
        if action == "A1":
            results.append(
                {
                    "VENTANA_ID": "NO REQUERIDA",
                    "VENTANA_INICIO_REAL": pd.NaT,
                    "VENTANA_FIN_REAL": pd.NaT,
                    "ESTADO_RVO": "MONITOREO",
                    "TIPO_VENTANA_RVO": "Operación normal",
                    "RESTRICCIONES_RVO": "Mantener vigilancia reforzada.",
                    "HOLGURA_DIAS": np.nan,
                }
            )
            continue
        if action == "A3":
            results.append(
                {
                    "VENTANA_ID": "NO REQUERIDA",
                    "VENTANA_INICIO_REAL": row["FECHA_DISPONIBLE_DESDE"],
                    "VENTANA_FIN_REAL": pd.NaT,
                    "ESTADO_RVO": "ACCIÓN OPERATIVA",
                    "TIPO_VENTANA_RVO": "Mitigación operacional",
                    "RESTRICCIONES_RVO": "Coordinar modificación temporal con Operaciones/Procesos.",
                    "HOLGURA_DIAS": np.nan,
                }
            )
            continue

        candidates = window_table[
            (window_table["ID_tramo"] == row["ID_tramo"])
            & (window_table["Inicio"] >= row["FECHA_DISPONIBLE_DESDE"])
            & (window_table["Inicio"] <= row["FECHA_LIMITE_PLANIFICACION"])
        ].sort_values("Inicio")

        if len(candidates):
            w = candidates.iloc[0]
            holgura = (row["FECHA_LIMITE_PLANIFICACION"] - w["Inicio"]).days
            results.append(
                {
                    "VENTANA_ID": w["ID_ventana"],
                    "VENTANA_INICIO_REAL": w["Inicio"],
                    "VENTANA_FIN_REAL": w["Fin"],
                    "ESTADO_RVO": "VENTANA FACTIBLE",
                    "TIPO_VENTANA_RVO": w["Tipo"],
                    "RESTRICCIONES_RVO": w["Restricciones"],
                    "HOLGURA_DIAS": holgura,
                }
            )
        else:
            results.append(
                {
                    "VENTANA_ID": "SIN VENTANA",
                    "VENTANA_INICIO_REAL": pd.NaT,
                    "VENTANA_FIN_REAL": pd.NaT,
                    "ESTADO_RVO": "SIN VENTANA FACTIBLE",
                    "TIPO_VENTANA_RVO": "REQUIERE GESTIÓN",
                    "RESTRICCIONES_RVO": "Gestionar una ventana operativa o coordinación extraordinaria.",
                    "HOLGURA_DIAS": np.nan,
                }
            )

    interventions = pd.concat([interventions.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    interventions["ORIGEN_VENTANA"] = np.where(
        interventions["TIPO_VENTANA_RVO"].eq("Ventana provisional calculada por el DSS"),
        "Calculada por DSS: fecha de corte + lead time; pendiente de confirmación operativa",
        np.where(
            interventions["VENTANA_ID"].isin(["NO REQUERIDA", "SIN VENTANA"]),
            interventions["TIPO_VENTANA_RVO"],
            "Ventana operativa ingresada/confirmada en la tabla de ventanas",
        ),
    )
    return interventions, policy_table


# =============================================================================
# 14. COSTEO TÉCNICO-OPERATIVO DEL CÓDIGO BASE
# =============================================================================

def _factor_diametro(d: float) -> float:
    if pd.isna(d):
        return 1.00
    if d <= 6:
        return 1.00
    if d <= 8:
        return 1.08
    if d <= 10:
        return 1.16
    return 1.25


def costeo(interventions: pd.DataFrame) -> pd.DataFrame:
    x = interventions.copy()
    if x.empty:
        return x

    x["FACTOR_DIAMETRO"] = x["Diametro_in"].apply(_factor_diametro)
    x["FACTOR_RIESGO"] = x["NIVEL_RIESGO"].map({"BAJO": 1.00, "MEDIO": 1.05, "ALTO": 1.15, "CRITICO": 1.25})
    x["FACTOR_SERVICIO"] = x["ID_tramo"].map(FACTOR_SERVICIO).fillna(1.10)
    x["FACTOR_COMPLEJIDAD"] = (x["FACTOR_DIAMETRO"] * x["FACTOR_RIESGO"] * x["FACTOR_SERVICIO"]).clip(1.0, 1.60)

    def calcular_duracion(row: pd.Series) -> int:
        base = BASE_RECURSOS[row["ACCION_MDP_ID"]]["duracion_h"]
        duracion = base * (0.70 + 0.30 * row["FACTOR_COMPLEJIDAD"])
        return int(math.ceil(duracion / 2) * 2)

    x["HORAS_INTERVENCION"] = x.apply(calcular_duracion, axis=1)

    for esp in ["mecanico", "soldador", "inspector", "corrosion", "planificador", "operaciones"]:
        col = "HH_" + esp.upper()
        x[col] = x.apply(
            lambda row: round(BASE_RECURSOS[row["ACCION_MDP_ID"]][f"hh_{esp}"] * row["FACTOR_COMPLEJIDAD"]),
            axis=1,
        ).astype(int)

    hh_cols = list(TARIFA_HH.keys())
    x["HH_TOTAL"] = x[hh_cols].sum(axis=1)
    x["PERSONAL_PROMEDIO_INTERVENCION"] = np.ceil(x["HH_TOTAL"] / x["HORAS_INTERVENCION"]).astype(int)

    for col, tarifa in TARIFA_HH.items():
        x["COSTO_" + col] = x[col] * tarifa
    x["COSTO_MANO_OBRA_S"] = x[["COSTO_" + c for c in TARIFA_HH]].sum(axis=1)
    x["COSTO_MATERIALES_S"] = x.apply(
        lambda row: MATERIAL_BASE[row["ACCION_MDP_ID"]] * row["FACTOR_DIAMETRO"] * row["FACTOR_SERVICIO"], axis=1
    ).round(0)
    x["COSTO_EQUIPOS_S"] = x.apply(
        lambda row: EQUIPO_BASE[row["ACCION_MDP_ID"]] * row["FACTOR_COMPLEJIDAD"], axis=1
    ).round(0)
    x["COSTO_SERVICIOS_TERCEROS_S"] = x.apply(
        lambda row: SERVICIO_BASE[row["ACCION_MDP_ID"]] * row["FACTOR_RIESGO"], axis=1
    ).round(0)
    x["COSTO_HSE_LOGISTICA_S"] = x["ACCION_MDP_ID"].map(COSTO_HSE_BASE)
    x["COSTO_DIRECTO_S"] = (
        x["COSTO_MANO_OBRA_S"]
        + x["COSTO_MATERIALES_S"]
        + x["COSTO_EQUIPOS_S"]
        + x["COSTO_SERVICIOS_TERCEROS_S"]
        + x["COSTO_HSE_LOGISTICA_S"]
    )
    x["COSTO_INDIRECTO_S"] = (x["COSTO_DIRECTO_S"] * 0.10).round(0)
    x["CONTINGENCIA_S"] = ((x["COSTO_DIRECTO_S"] + x["COSTO_INDIRECTO_S"]) * 0.08).round(0)
    x["PRESUPUESTO_MANTENIMIENTO_S"] = (x["COSTO_DIRECTO_S"] + x["COSTO_INDIRECTO_S"] + x["CONTINGENCIA_S"]).round(0)

    x["HORAS_INDISPONIBILIDAD_EST"] = x.apply(
        lambda row: row["HORAS_INTERVENCION"] * FACTOR_INDISPONIBILIDAD.get(row["REQUIERE_PARADA"], 0.5), axis=1
    ).round(1)
    x["COSTO_INDISPONIBILIDAD_H_S"] = x["ID_tramo"].map(COSTO_INDISPONIBILIDAD_H).fillna(3500)
    x["IMPACTO_OPERATIVO_S"] = (x["HORAS_INDISPONIBILIDAD_EST"] * x["COSTO_INDISPONIBILIDAD_H_S"]).round(0)
    x["COSTO_ECONOMICO_TOTAL_S"] = (x["PRESUPUESTO_MANTENIMIENTO_S"] + x["IMPACTO_OPERATIVO_S"]).round(0)
    x["HORAS_UT"] = x["ACCION_MDP_ID"].map(USO_UT)
    x["HORAS_EQUIPO_SOLDADURA"] = x["ACCION_MDP_ID"].map(USO_SOLDADURA)
    x["USA_PARADA"] = x["REQUIERE_PARADA"].isin(["SI", "PROBABLE"]).astype(int)
    return x


# =============================================================================
# 15. CAPA 6 - MILP CON SciPy/HiGHS
# =============================================================================
# Se utiliza scipy.optimize.milp para evitar el error observado en Colab:
#   TypeError: LpVariable.__init__() got an unexpected keyword argument 'cat'
# Ese error corresponde al backend/paquete PuLP disponible en el entorno, no a
# la formulación matemática. Aquí se conserva exactamente la misma estructura:
# - decisión binaria x[i,t]
# - binaria y[i] para no programado
# - asignación única
# - recursos por semana
# - UT y soldadura
# - máximo de frentes y paradas
# - presupuesto mensual y anual
# - misma función objetivo de retraso, espera, impacto y no programación.


def _build_calendar(fecha_inicio: pd.Timestamp, horizonte_semanas: int) -> pd.DataFrame:
    cal = pd.DataFrame({"Periodo": range(1, horizonte_semanas + 1)})
    cal["Fecha_inicio"] = [fecha_inicio + pd.Timedelta(days=7 * (t - 1)) for t in cal["Periodo"]]
    cal["Fecha_fin"] = cal["Fecha_inicio"] + pd.Timedelta(days=6)
    cal["MES"] = cal["Fecha_inicio"].dt.to_period("M").astype(str)
    return cal


def _prepare_milp_jobs(
    interventions: pd.DataFrame,
    horizonte_semanas: int = HORIZONTE_SEMANAS_DEFAULT,
    fecha_inicio: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, List[int]]]:
    jobs = costeo(interventions[interventions["INGRESA_MILP"].eq("SI")].copy()).reset_index(drop=True)
    if jobs.empty:
        return jobs, pd.DataFrame(), {}

    if fecha_inicio is None:
        fecha_inicio = pd.Timestamp(jobs["FECHA_DETECCION"].min()).normalize()
    fecha_inicio = pd.Timestamp(fecha_inicio).normalize()
    cal = _build_calendar(fecha_inicio, horizonte_semanas)

    jobs["LEAD_TIME_DIAS_MILP"] = jobs["ACCION_MDP_ID"].map(LEAD_TIME_DIAS)
    base_date = jobs["FECHA_CORTE_ANALISIS"] if "FECHA_CORTE_ANALISIS" in jobs.columns else jobs["FECHA_DETECCION"]
    jobs["FECHA_MAS_TEMPRANA"] = pd.to_datetime(base_date) + pd.to_timedelta(jobs["LEAD_TIME_DIAS_MILP"], unit="D")

    def periodo_fecha(fecha: pd.Timestamp) -> int:
        possibles = cal[(cal["Fecha_inicio"] <= fecha) & (cal["Fecha_fin"] >= fecha)]
        if len(possibles):
            return int(possibles.iloc[0]["Periodo"])
        if fecha < cal["Fecha_inicio"].min():
            return int(cal["Periodo"].min())
        return int(cal["Periodo"].max())

    jobs["PERIODO_LIMITE_RVO"] = jobs["FECHA_LIMITE_PLANIFICACION"].apply(periodo_fecha)
    factor_priority = {"P1 - URGENTE": 10, "P2 - PRIORITARIA": 6, "P3 - PROGRAMABLE": 3, "P4 - RUTINARIA": 1}
    jobs["FACTOR_PRIORIDAD_MILP"] = jobs["PRIORIDAD_MTTO"].map(factor_priority).fillna(2)

    allowed: Dict[int, List[int]] = {}
    for i, row in jobs.iterrows():
        if row["ESTADO_RVO"] == "VENTANA FACTIBLE" and pd.notna(row["VENTANA_INICIO_REAL"]):
            candidates = cal[
                (cal["Fecha_fin"] >= row["VENTANA_INICIO_REAL"])
                & (cal["Fecha_inicio"] <= row["VENTANA_FIN_REAL"])
            ]["Periodo"].astype(int).tolist()
        else:
            candidates = cal[cal["Fecha_fin"] >= row["FECHA_MAS_TEMPRANA"]]["Periodo"].astype(int).tolist()
        allowed[i] = candidates

    return jobs, cal, allowed


def _diagnose_unscheduled(row: pd.Series, allowed: Sequence[int], presupuesto_mensual: float) -> str:
    reasons: List[str] = []
    if len(allowed) == 0:
        reasons.append("sin semanas candidatas dentro del horizonte")
    if row["PRESUPUESTO_MANTENIMIENTO_S"] > presupuesto_mensual:
        reasons.append("costo del trabajo supera el presupuesto mensual")
    for resource, capacity in CAPACIDAD_SEMANAL.items():
        if row[resource] > capacity:
            reasons.append(f"{resource} supera capacidad semanal")
    if row["HORAS_UT"] > CAP_UT_H_SEMANA:
        reasons.append("horas UT superan capacidad semanal")
    if row["HORAS_EQUIPO_SOLDADURA"] > CAP_SOLDADURA_H_SEMANA:
        reasons.append("horas de soldadura superan capacidad semanal")
    if not reasons:
        reasons.append("conflicto combinado de presupuesto, recursos, ventana o prioridad con otros trabajos")
    return "; ".join(reasons).capitalize() + "."


def layer6_milp(
    interventions: pd.DataFrame,
    horizonte_semanas: int = HORIZONTE_SEMANAS_DEFAULT,
    presupuesto_mensual: float = PRESUPUESTO_MENSUAL_DEFAULT,
    fecha_inicio: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    if presupuesto_mensual < 0:
        raise ValueError("El presupuesto mensual no puede ser negativo.")
    if horizonte_semanas < 1:
        raise ValueError("El horizonte de programación debe tener al menos una semana.")

    jobs, cal, allowed = _prepare_milp_jobs(interventions, horizonte_semanas, fecha_inicio)
    if jobs.empty:
        return jobs.assign(
            ESTADO_PROGRAMACION=pd.Series(dtype=str),
            SEMANA_PROGRAMADA=pd.Series(dtype=float),
            FECHA_PROGRAMADA=pd.Series(dtype="datetime64[ns]"),
            FECHA_FIN_SEMANA=pd.Series(dtype="datetime64[ns]"),
            CUMPLIMIENTO_RVO=pd.Series(dtype=str),
            MOTIVO_NO_PROGRAMACION=pd.Series(dtype=str),
            SOLVER_MILP=pd.Series(dtype=str),
            ESTADO_SOLVER=pd.Series(dtype=str),
        )

    x_keys = [(i, t) for i in jobs.index for t in allowed[i]]
    y_keys = list(jobs.index)
    index: Dict[tuple, int] = {("x", i, t): k for k, (i, t) in enumerate(x_keys)}
    offset = len(x_keys)
    index.update({("y", i): offset + k for k, i in enumerate(y_keys)})
    n_vars = len(index)

    # -------------------------------------------------------------------------
    # Función objetivo exacta del código base.
    # -------------------------------------------------------------------------
    c = np.zeros(n_vars, dtype=float)
    for i in jobs.index:
        limit_period = jobs.loc[i, "PERIODO_LIMITE_RVO"]
        priority_factor = jobs.loc[i, "FACTOR_PRIORIDAD_MILP"]
        risk = jobs.loc[i, "Riesgo_DRBI"]
        impact = jobs.loc[i, "IMPACTO_OPERATIVO_S"]
        if allowed[i]:
            first_allowed = min(allowed[i])
            for t in allowed[i]:
                delay = max(0, t - limit_period)
                wait = max(0, t - first_allowed)
                c[index[("x", i, t)]] = (
                    delay * priority_factor * (10000 + 10000 * risk)
                    + wait * priority_factor * 500
                    + 0.001 * impact
                )
        c[index[("y", i)]] = 1000000 * priority_factor * (1 + risk)

    rows: List[Dict[int, float]] = []
    lb: List[float] = []
    ub: List[float] = []

    def add_constraint(coeffs: Dict[int, float], low: float = -np.inf, high: float = np.inf) -> None:
        rows.append(coeffs)
        lb.append(low)
        ub.append(high)

    # 1) Cada intervención se programa una sola vez o queda no programada.
    for i in jobs.index:
        coeff = {index[("x", i, t)]: 1.0 for t in allowed[i]}
        coeff[index[("y", i)]] = 1.0
        add_constraint(coeff, 1.0, 1.0)

    # 2) Capacidades semanales.
    for t in cal["Periodo"]:
        t = int(t)
        for resource, capacity in CAPACIDAD_SEMANAL.items():
            add_constraint(
                {index[("x", i, t)]: float(jobs.loc[i, resource]) for i in jobs.index if ("x", i, t) in index},
                high=float(capacity),
            )
        add_constraint(
            {index[("x", i, t)]: float(jobs.loc[i, "HORAS_UT"]) for i in jobs.index if ("x", i, t) in index},
            high=float(CAP_UT_H_SEMANA),
        )
        add_constraint(
            {index[("x", i, t)]: float(jobs.loc[i, "HORAS_EQUIPO_SOLDADURA"]) for i in jobs.index if ("x", i, t) in index},
            high=float(CAP_SOLDADURA_H_SEMANA),
        )
        add_constraint(
            {index[("x", i, t)]: 1.0 for i in jobs.index if ("x", i, t) in index},
            high=float(MAX_FRENTES_SEMANA),
        )
        add_constraint(
            {index[("x", i, t)]: float(jobs.loc[i, "USA_PARADA"]) for i in jobs.index if ("x", i, t) in index},
            high=float(MAX_PARADAS_SIMULTANEAS),
        )

    # 3) Presupuesto mensual.
    for mes in cal["MES"].unique():
        weeks = set(cal.loc[cal["MES"].eq(mes), "Periodo"].astype(int).tolist())
        add_constraint(
            {
                index[("x", i, t)]: float(jobs.loc[i, "PRESUPUESTO_MANTENIMIENTO_S"])
                for i in jobs.index
                for t in allowed[i]
                if t in weeks
            },
            high=float(presupuesto_mensual),
        )

    # 4) Presupuesto anual equivalente al código base.
    add_constraint(
        {index[("x", i, t)]: float(jobs.loc[i, "PRESUPUESTO_MANTENIMIENTO_S"]) for i, t in x_keys},
        high=float(presupuesto_mensual * 12),
    )

    A = lil_matrix((len(rows), n_vars), dtype=float)
    for r, coeffs in enumerate(rows):
        for j, value in coeffs.items():
            A[r, j] = value

    result = milp(
        c=c,
        integrality=np.ones(n_vars, dtype=int),
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=LinearConstraint(A.tocsr(), np.asarray(lb), np.asarray(ub)),
        options={"time_limit": 60.0, "mip_rel_gap": 0.001},
    )

    if result.x is None:
        solution = np.zeros(n_vars)
    else:
        solution = result.x

    solver_status = str(getattr(result, "message", "Sin mensaje del solver"))
    scheduled_rows: List[tuple] = []
    for i, row in jobs.iterrows():
        programmed_period = None
        for t in allowed[i]:
            if solution[index[("x", i, t)]] > 0.5:
                programmed_period = t
                break

        if programmed_period is None:
            scheduled_rows.append(
                (
                    "NO PROGRAMADO",
                    np.nan,
                    pd.NaT,
                    pd.NaT,
                    "PENDIENTE DE PROGRAMACIÓN",
                    _diagnose_unscheduled(row, allowed[i], presupuesto_mensual),
                )
            )
            continue

        cal_row = cal[cal["Periodo"].eq(programmed_period)].iloc[0]

        # Fecha exacta propuesta dentro del periodo MILP. No se utiliza de forma
        # automática el inicio de la semana: la fecha debe respetar la fecha más
        # temprana por lead time y, cuando existe, el inicio de la ventana RVO.
        execution_date = max(pd.Timestamp(cal_row["Fecha_inicio"]), pd.Timestamp(row["FECHA_MAS_TEMPRANA"]))
        if pd.notna(row.get("VENTANA_INICIO_REAL", pd.NaT)):
            execution_date = max(execution_date, pd.Timestamp(row["VENTANA_INICIO_REAL"]))
        execution_end = execution_date + pd.Timedelta(hours=float(row.get("HORAS_INTERVENCION", 0)))

        compliance = (
            "DENTRO DE PLAZO RVO"
            if execution_date.normalize() <= pd.Timestamp(row["FECHA_LIMITE_PLANIFICACION"]).normalize()
            else "FUERA DE PLAZO RVO"
        )

        if pd.notna(row.get("VENTANA_INICIO_REAL", pd.NaT)):
            basis = (
                f"MILP seleccionó el periodo {int(programmed_period)}; fecha exacta = máximo entre inicio del periodo, "
                "fecha mínima por lead time e inicio de la ventana RVO."
            )
        else:
            basis = (
                f"MILP seleccionó el periodo {int(programmed_period)}; fecha exacta = máximo entre inicio del periodo "
                "y fecha mínima por lead time. No existe ventana física confirmada."
            )

        scheduled_rows.append(
            (
                "PROGRAMADO",
                programmed_period,
                execution_date,
                execution_end,
                compliance,
                basis,
            )
        )

    jobs[
        [
            "ESTADO_PROGRAMACION",
            "SEMANA_PROGRAMADA",
            "FECHA_PROGRAMADA",
            "FECHA_FIN_SEMANA",
            "CUMPLIMIENTO_RVO",
            "MOTIVO_NO_PROGRAMACION",
        ]
    ] = pd.DataFrame(scheduled_rows, index=jobs.index)
    jobs["SOLVER_MILP"] = "SciPy/HiGHS"
    jobs["ESTADO_SOLVER"] = solver_status

    return jobs


# =============================================================================
# 16. CAPA 7 - HORIZONTE RODANTE / REPROGRAMACIÓN
# =============================================================================

def rolling_horizon(
    programa: pd.DataFrame,
    df_drbi: pd.DataFrame,
    semana_revision: Optional[int] = None,
    ventana_congelada_semanas: int = VENTANA_CONGELADA_SEMANAS_DEFAULT,
    umbral_incremento_riesgo: float = UMBRAL_INCREMENTO_RIESGO_DEFAULT,
) -> pd.DataFrame:
    if programa is None or programa.empty:
        return pd.DataFrame()

    if semana_revision is None or int(semana_revision) <= 0:
        semana_revision = int(pd.to_numeric(df_drbi["Periodo_semana"], errors="coerce").max())
    else:
        semana_revision = int(semana_revision)

    hist = df_drbi[pd.to_numeric(df_drbi["Periodo_semana"], errors="coerce") <= semana_revision].copy()
    hist = hist.sort_values(["ID_tramo", "Periodo_semana"])
    if hist.empty:
        raise ValueError("La semana de revisión seleccionada es anterior a todos los registros disponibles.")

    actual = (
        hist.groupby("ID_tramo").tail(1)[["ID_tramo", "Periodo_semana", "PoF_posterior", "Riesgo_DRBI", "NIVEL_RIESGO"]]
        .rename(
            columns={
                "Periodo_semana": "SEMANA_RIESGO_ACTUAL",
                "PoF_posterior": "PoF_ACTUAL",
                "Riesgo_DRBI": "RIESGO_ACTUAL",
                "NIVEL_RIESGO": "NIVEL_RIESGO_ACTUAL",
            }
        )
    )

    prev_hist = df_drbi[pd.to_numeric(df_drbi["Periodo_semana"], errors="coerce") < semana_revision].copy()
    prev_hist = prev_hist.sort_values(["ID_tramo", "Periodo_semana"])
    if len(prev_hist):
        anterior = prev_hist.groupby("ID_tramo").tail(1)[["ID_tramo", "Riesgo_DRBI"]].rename(columns={"Riesgo_DRBI": "RIESGO_ANTERIOR"})
    else:
        anterior = pd.DataFrame(columns=["ID_tramo", "RIESGO_ANTERIOR"])

    revision = actual.merge(anterior, on="ID_tramo", how="left")
    revision["RIESGO_ANTERIOR"] = revision["RIESGO_ANTERIOR"].fillna(revision["RIESGO_ACTUAL"])
    revision["DELTA_RIESGO"] = (revision["RIESGO_ACTUAL"] - revision["RIESGO_ANTERIOR"]).round(4)
    revision["VARIACION_RIESGO_PCT"] = np.where(
        revision["RIESGO_ANTERIOR"] > 0,
        revision["DELTA_RIESGO"] / revision["RIESGO_ANTERIOR"] * 100,
        0,
    ).round(2)

    plan = programa.merge(revision, on="ID_tramo", how="left")
    plan["SEMANAS_HASTA_PROGRAMACION"] = pd.to_numeric(plan["SEMANA_PROGRAMADA"], errors="coerce") - semana_revision

    def clasificar_horizonte(row: pd.Series) -> str:
        s = row.get("SEMANA_PROGRAMADA")
        if pd.isna(s):
            return "SIN PROGRAMAR"
        diferencia = float(s) - semana_revision
        if diferencia < 0:
            return "REQUIERE CONFIRMAR EJECUCIÓN"
        if diferencia <= ventana_congelada_semanas:
            return "VENTANA CONGELADA"
        return "REOPTIMIZABLE"

    plan["ESTADO_HORIZONTE"] = plan.apply(clasificar_horizonte, axis=1)
    plan["ALERTA_INCREMENTO_RIESGO"] = np.where(plan["DELTA_RIESGO"] >= umbral_incremento_riesgo, "SI", "NO")
    plan["JERARQUIA_ACTUAL"] = plan["NIVEL_RIESGO_ACTUAL"].map(JERARQUIA_RIESGO)
    plan["JERARQUIA_PLAN"] = plan["NIVEL_RIESGO"].map(JERARQUIA_RIESGO)
    plan["RIESGO_ESCALADO"] = np.where(plan["JERARQUIA_ACTUAL"] > plan["JERARQUIA_PLAN"], "SI", "NO")

    def decidir(row: pd.Series) -> str:
        if row.get("ESTADO_PROGRAMACION") == "NO PROGRAMADO":
            return "REOPTIMIZAR - SIN PROGRAMACIÓN"
        if row.get("RIESGO_ESCALADO") == "SI":
            return "REOPTIMIZAR - RIESGO ESCALADO"
        if row.get("ALERTA_INCREMENTO_RIESGO") == "SI":
            return "REOPTIMIZAR - INCREMENTO DE RIESGO"
        if row.get("CUMPLIMIENTO_RVO") == "FUERA DE PLAZO RVO":
            return "REOPTIMIZAR - PLAZO RVO"
        if row.get("ESTADO_HORIZONTE") == "VENTANA CONGELADA":
            return "MANTENER PROGRAMACIÓN"
        if row.get("ESTADO_HORIZONTE") == "REOPTIMIZABLE":
            return "MANTENER / EVALUAR EN PRÓXIMO CICLO"
        return "REVISAR ESTADO"

    plan["DECISION_ROLLING_HORIZON"] = plan.apply(decidir, axis=1)

    def accion_planificacion(row: pd.Series) -> str:
        decision = row["DECISION_ROLLING_HORIZON"]
        if "REOPTIMIZAR" in decision:
            return "Enviar nuevamente a MDP-RVO-MILP."
        if decision == "MANTENER PROGRAMACIÓN":
            return "Congelar fecha, reservar recursos y preparar orden de trabajo."
        if decision == "MANTENER / EVALUAR EN PRÓXIMO CICLO":
            return "Mantener fecha provisional y reevaluar en la siguiente revisión."
        return "Validar ejecución con Mantenimiento y actualizar estado."

    plan["ACCION_PLANIFICACION"] = plan.apply(accion_planificacion, axis=1)
    return plan


# =============================================================================
# 17. POLÍTICAS DE MANTENIMIENTO Y PROGRAMACIÓN
# =============================================================================

def maintenance_policy(tipo: str) -> pd.DataFrame:
    """Matriz general de política de mantenimiento.

    La política NO reemplaza el cálculo MDP-RVO: lo explica y lo vuelve
    operacional. Evalúa conjuntamente riesgo DRBI, evidencia IOW, cambio de PoF,
    integridad física, variables químicas/operativas, consecuencia de falla y
    restricciones propias del tipo de tubería.
    """
    if tipo not in TIPOS_TUBERIA:
        raise ValueError(f"Tipo de tubería no válido: {tipo}")

    type_context = {
        "Submarina": (
            "Confirmar condición externa/interna, acceso submarino, ventana metoceánica, ROV/buceo, NDE, logística marina, "
            "protección externa y disponibilidad de repuestos. Una condición de alto riesgo no debe postergarse solo por dificultad de acceso."
        ),
        "Transferencia": (
            "Verificar continuidad de despacho/recepción, inventarios, líneas alternas, posibilidad de aislamiento, drenaje y retorno al servicio. "
            "Evitar indisponibilidades simultáneas que comprometan transferencia de crudo o productos."
        ),
        "Proceso": (
            "Coordinar con Operaciones de la unidad, aislamiento, despresurización, permisos, IOWs, carga de proceso y oportunidad de parada. "
            "Priorizar desviaciones persistentes, pérdida de espesor, incremento de corrosión y servicios de mayor consecuencia."
        ),
    }[tipo]

    common_variables = (
        "IOWs desviadas; magnitud M(t); persistencia D(t); frecuencia F(t); evidencia E(t); PoF inicial; PoF actualizada; ΔPoF; "
        "CoF; riesgo DRBI; espesor actual; velocidad de corrosión; vida remanente; pH; cloruros; salinidad; TAN; hierro disuelto; "
        "BSW; temperatura; presión; inhibidor; servicio/fluido; diámetro; longitud; tendencia temporal y disponibilidad de ventana/recursos."
    )

    rows = [
        {
            "Nivel de riesgo": "BAJO",
            "Rango DRBI": "Riesgo < 0.25",
            "Qué debe observarse": "Evidencia IOW baja/controlada, PoF estable, sin pérdida acelerada de espesor y vida remanente holgada.",
            "Acción base": "A1 - Vigilancia de condición reforzada; A2 si existe incertidumbre relevante de integridad.",
            "Por qué se ejecuta así": (
                "El riesgo actual no justifica indisponibilidad innecesaria. Se conserva la operación, pero se aumenta vigilancia para detectar "
                "cambios tempranos. Si la PoF sube, aparecen desviaciones persistentes o la condición física no es confiable, se escala a inspección."
            ),
            "Plazo RVO de referencia": "Hasta 180 días, limitado adicionalmente por vida remanente y nueva evidencia.",
            "Condiciones que obligan a escalar": "Aumento de nivel DRBI; ΔPoF significativo; E(t) creciente; RUL decreciente; CR creciente; nueva IOW persistente; CoF elevada.",
            "Validación antes de cerrar": "Tendencia estable, evidencia controlada, datos completos y sin escalamiento de riesgo.",
            "Responsables": "Operaciones / Corrosión / Integridad.",
        },
        {
            "Nivel de riesgo": "MEDIO",
            "Rango DRBI": "0.25 ≤ Riesgo < 0.60",
            "Qué debe observarse": "Una o más variables muestran degradación suficiente para elevar el riesgo o la incertidumbre de integridad.",
            "Acción base": "A2 - Inspección dirigida y/o A3 - Mitigación operativa; aplicar la acción seleccionada por MDP.",
            "Por qué se ejecuta así": (
                "El objetivo es reducir incertidumbre y frenar degradación antes de entrar a nivel alto. La inspección verifica espesor/defecto; "
                "la mitigación reduce exposición si las IOWs o variables químicas/operativas están impulsando la corrosión."
            ),
            "Plazo RVO de referencia": "Hasta 90 días, sujeto al plazo calculado por RVO.",
            "Condiciones que obligan a escalar": "PoF o DRBI ascendentes; varias IOWs fuera de límite; persistencia/frecuencia crecientes; vida remanente corta; CoF alta.",
            "Validación antes de cerrar": "Inspección/mitigación registrada, condición verificada y riesgo recalculado.",
            "Responsables": "Integridad / Inspección / Corrosión / Operaciones / Mantenimiento.",
        },
        {
            "Nivel de riesgo": "ALTO",
            "Rango DRBI": "0.60 ≤ Riesgo < 1.00",
            "Qué debe observarse": "PoF y/o CoF suficientes para requerir intervención prioritaria, normalmente acompañadas por tendencia de degradación o evidencia persistente.",
            "Acción base": "A4 - Mantenimiento basado en condición o A5 - Preventivo planificado, según la política MDP.",
            "Por qué se ejecuta así": (
                "Diferir aumenta exposición a parada no programada. La fecha se define antes del límite RVO y la planificación debe reservar recursos, "
                "aislamiento/ventana y presupuesto. Si la condición empeora, se reoptimiza sin esperar al siguiente plan ordinario."
            ),
            "Plazo RVO de referencia": "Hasta 30 días o menos si vida remanente/ventana obligan a adelantar.",
            "Condiciones que obligan a escalar": "Riesgo llega a CRÍTICO; RUL menor al horizonte de ejecución; espesor cercano a mínimo; incremento brusco de CR/PoF; pérdida de ventana segura.",
            "Validación antes de cerrar": "Trabajo ejecutado o control temporal formal; inspección posintervención; nuevo cálculo Bayes/DRBI.",
            "Responsables": "Integridad / Mantenimiento / Planificación / Operaciones / SSOMA.",
        },
        {
            "Nivel de riesgo": "CRITICO",
            "Rango DRBI": "Riesgo ≥ 1.00",
            "Qué debe observarse": "Exposición no tolerable del modelo: alta probabilidad, alta consecuencia o combinación de ambas con evidencia de degradación.",
            "Acción base": "A6 - Intervención correctiva prioritaria de integridad; control temporal formal únicamente mientras se ejecuta la solución.",
            "Por qué se ejecuta así": (
                "La prioridad es reducir inmediatamente la exposición a pérdida de contención/parada no programada. Producción, presupuesto o conveniencia "
                "de calendario no deben ser la única razón para diferir una condición crítica; cualquier excepción requiere gestión de integridad formal."
            ),
            "Plazo RVO de referencia": "Hasta 7 días o el menor plazo seguro determinado por Integridad/Operaciones.",
            "Condiciones que obligan a escalar": "Sin ventana factible; condición física no aceptable; deterioro acelerado; imposibilidad de mitigar; recursos críticos no disponibles.",
            "Validación antes de cerrar": "Integridad restablecida/documentada, controles retirados de forma segura y riesgo reducido tras nueva evaluación.",
            "Responsables": "Integridad / Operaciones / Mantenimiento / Planificación / SSOMA / Gerencia operativa según criticidad.",
        },
    ]
    df = pd.DataFrame(rows)
    df["Variables evaluadas por la política"] = common_variables
    df["Adaptación al tipo de tubería"] = type_context
    return df


def scheduling_policy(tipo: str) -> pd.DataFrame:
    """Reglas de programación que explican por qué MILP/Rolling Horizon agenda o difiere un trabajo."""
    if tipo not in TIPOS_TUBERIA:
        raise ValueError(f"Tipo de tubería no válido: {tipo}")

    adaptation = {
        "Submarina": "Incluir disponibilidad de embarcación/ROV/buceo/NDE, condiciones marinas, logística y accesibilidad del tramo.",
        "Transferencia": "Alinear con inventario, despacho/recepción, línea alterna, drenaje, aislamiento y continuidad logística.",
        "Proceso": "Alinear con carga de unidad, permisos, aislamiento, despresurización, coordinación con Operaciones y ventanas de baja carga/parada.",
    }[tipo]

    rows = [
        (1, "Prioridad por riesgo", "Riesgo DRBI, nivel de riesgo, PoF actualizada, ΔPoF y CoF.", "Un trabajo de mayor exposición debe competir con mayor peso que uno de menor riesgo.", "Priorizar primero CRÍTICO/ALTO y luego MEDIO, manteniendo trazabilidad de excepciones."),
        (2, "Límite RVO", "Fecha límite RVO, urgencia, lead time y vida remanente.", "La programación debe ocurrir antes del último momento técnicamente seguro definido por el modelo.", "No programar después del límite salvo decisión formal de gestión de integridad y control compensatorio."),
        (3, "Ventana operativa", "Requiere parada/aislamiento, ventana disponible, condiciones operativas y tipo de intervención.", "Una fecha solo es ejecutable si puede aislarse/intervenirse el activo de forma segura.", "Usar primera ventana factible compatible; si no existe, gestionar ventana extraordinaria o mitigación temporal."),
        (4, "Recursos", "HH por especialidad, UT/NDE, soldadura, frentes simultáneos y capacidad semanal.", "Evita planes matemáticamente atractivos pero imposibles de ejecutar por falta de capacidad.", "Asignar semana que respete capacidad; el trabajo permanece visible si no cabe."),
        (5, "Presupuesto", "Costo de mantenimiento, servicios, materiales, HSE/logística y presupuesto mensual.", "Permite secuenciar el portafolio sin eliminar trabajos requeridos.", "Si el presupuesto no alcanza, dejar como pendiente con causa explícita y escalar si compromete RVO."),
        (6, "Impacto operacional", "Horas de intervención, necesidad de parada, impacto económico y servicio/fluido.", "Busca reducir afectación productiva sin sacrificar la prioridad de integridad.", "Aprovechar baja carga/parada y agrupar cuando sea seguro, pero no desplazar riesgo crítico por conveniencia."),
        (7, "Horizonte congelado", "Semana programada, recursos reservados y proximidad de ejecución.", "Evita reprogramación continua de trabajos ya preparados.", "Congelar próximas 4 semanas salvo escalamiento de riesgo, pérdida de ventana o imposibilidad material."),
        (8, "Rolling Horizon", "Riesgo actual vs. previo, cambio de nivel, Δriesgo, nueva evidencia y estado del plan.", "La corrosión y la operación cambian; el programa debe adaptarse cuando la evidencia cambia.", "Reoptimizar si sube el nivel, Δriesgo ≥ umbral, cambia el RVO o el trabajo queda sin programación."),
        (9, "Calidad de datos", "Completitud, duplicados, interpolaciones, consistencia temporal y validez de variables.", "Una decisión de mantenimiento no debe parecer precisa si la evidencia de entrada no es confiable.", "Bloquear o marcar para revisión cuando la calidad no permite sustentar la decisión."),
        (10, "Trazabilidad y cierre", "Causa, acción, fecha, recursos, costo, resultado de inspección/intervención y KPI.", "Permite auditar por qué se tomó la decisión y retroalimentar Bayes/DRBI.", "Cerrar solo con evidencia de ejecución y nueva condición registrada; alimentar el siguiente ciclo semanal."),
    ]
    df = pd.DataFrame(rows, columns=["Orden", "Regla", "Variables que revisa", "Por qué se controla", "Decisión de programación"])
    df["Adaptación al tipo de tubería"] = adaptation
    return df


def policy_decision_detail(result: dict, tipo: str) -> pd.DataFrame:
    """Explica, tramo por tramo, por qué el DSS propone la acción vigente.

    Esta tabla es de interfaz/validación y NO se exporta al reporte Excel.
    """
    if tipo not in TIPOS_TUBERIA:
        raise ValueError(f"Tipo de tubería no válido: {tipo}")

    l4 = _latest(result["capa4"]) if "capa4" in result else pd.DataFrame()
    interventions = result.get("intervenciones", pd.DataFrame()).copy()
    program = result.get("programa", pd.DataFrame()).copy()
    rows = []

    for _, row in l4.iterrows():
        tid = str(row["ID_tramo"])
        iv = interventions[interventions["ID_tramo"].astype(str).eq(tid)] if not interventions.empty else pd.DataFrame()
        pr = program[program["ID_tramo"].astype(str).eq(tid)] if not program.empty else pd.DataFrame()
        ivr = iv.iloc[0] if len(iv) else None
        prr = pr.iloc[0] if len(pr) else None

        deviated = []
        for code, label in IOW_NOMBRES.items():
            col = f"DEV_{code}"
            if col in row.index and bool(row[col]):
                deviated.append(label)
        dev_text = ", ".join(deviated) if deviated else "Sin IOW fuera de límite en el estado actual"

        drivers = []
        if float(row.get("E_t", 0) or 0) >= 0.25:
            drivers.append(f"evidencia IOW E(t)={float(row.get('E_t')):.3f}")
        if float(row.get("Delta_PoF", 0) or 0) > 0:
            drivers.append(f"PoF aumentó Δ={float(row.get('Delta_PoF')):.4f}")
        if float(row.get("Velocidad_corrosion_mpy", 0) or 0) >= 3:
            drivers.append(f"velocidad de corrosión={float(row.get('Velocidad_corrosion_mpy')):.2f} mpy")
        if float(row.get("Vida_remanente_anios", 999) or 999) <= 5:
            drivers.append(f"vida remanente={float(row.get('Vida_remanente_anios')):.2f} años")
        if float(row.get("CoF_valor", 0) or 0) >= 4:
            drivers.append(f"CoF elevada={float(row.get('CoF_valor')):.1f}")
        if deviated:
            drivers.append(f"IOWs desviadas: {dev_text}")
        if not drivers:
            drivers.append("condición actual estable respecto de los umbrales del modelo")

        action = ivr["ACCION_MDP_ID"] if ivr is not None else "A1"
        action_name = ivr["ESTRATEGIA_MANTENIMIENTO"] if ivr is not None else ACCIONES["A1"]["nombre"]
        priority = ivr["PRIORIDAD_MTTO"] if ivr is not None else "SEGUIMIENTO"
        rvo = ivr["FECHA_LIMITE_PLANIFICACION"] if ivr is not None else pd.NaT
        window_state = ivr["ESTADO_RVO"] if ivr is not None else "NO REQUERIDA"
        program_state = prr["ESTADO_PROGRAMACION"] if prr is not None else "NO REQUIERE PROGRAMACIÓN"
        program_date = prr["FECHA_PROGRAMADA"] if prr is not None else pd.NaT
        program_reason = prr["EXPLICACION_PROGRAMACION"] if prr is not None and "EXPLICACION_PROGRAMACION" in prr.index else "Seguimiento por condición; no ingresa al MILP."

        rationale = (
            f"El tramo presenta riesgo DRBI={float(row['Riesgo_DRBI']):.4f} ({row['NIVEL_RIESGO']}), "
            f"PoF {float(row['PoF_prior']):.4f}→{float(row['PoF_posterior']):.4f} y CoF={float(row['CoF_valor']):.1f}. "
            f"Factores que sustentan la atención: {'; '.join(drivers)}. Por ello el MDP-RVO propone {action} - {action_name}."
        )

        rows.append({
            "Tramo": tid,
            "Tipo": tipo,
            "Servicio / fluido": row.get("Servicio_fluido", ""),
            "Nivel de riesgo": row.get("NIVEL_RIESGO", ""),
            "Riesgo DRBI": round(float(row.get("Riesgo_DRBI", 0)), 4),
            "PoF inicial": round(float(row.get("PoF_prior", 0)), 4),
            "PoF actualizada": round(float(row.get("PoF_posterior", 0)), 4),
            "ΔPoF": round(float(row.get("Delta_PoF", 0)), 4),
            "CoF": round(float(row.get("CoF_valor", 0)), 3),
            "IOWs desviadas": int(row.get("N_IOW_DESVIADAS", 0)),
            "Magnitud M(t)": round(float(row.get("M_t", 0)), 4),
            "Persistencia D(t)": round(float(row.get("D_t", 0)), 4),
            "Frecuencia F(t)": round(float(row.get("F_t", 0)), 4),
            "Evidencia E(t)": round(float(row.get("E_t", 0)), 4),
            "Variables IOW actualmente desviadas": dev_text,
            "Espesor actual (mm)": round(float(row.get("Espesor_actual_mm", 0)), 3),
            "Velocidad de corrosión (mpy)": round(float(row.get("Velocidad_corrosion_mpy", 0)), 3),
            "Vida remanente (años)": round(float(row.get("Vida_remanente_anios", 0)), 2),
            "pH": round(float(row.get("pH", 0)), 3),
            "Cloruros (ppm)": round(float(row.get("Cloruros_ppm", 0)), 2),
            "Salinidad (PTB)": round(float(row.get("Salinidad_PTB", 0)), 2),
            "TAN (mg KOH/g)": round(float(row.get("TAN_mgKOH_g", 0)), 3),
            "Hierro disuelto (mg/L)": round(float(row.get("Hierro_mg_L", 0)), 3),
            "BSW (%)": round(float(row.get("BSW_pct", 0)), 3),
            "Temperatura (°C)": round(float(row.get("Temperatura_C", 0)), 2),
            "Presión (bar g)": round(float(row.get("Presion_bar_g", 0)), 2),
            "Inhibidor (ppm)": round(float(row.get("Inhibidor_ppm", 0)), 2),
            "Acción propuesta": f"{action} - {action_name}",
            "Prioridad": priority,
            "Fecha límite RVO": rvo,
            "Estado de ventana": window_state,
            "Estado de programación": program_state,
            "Fecha programada": program_date,
            "Por qué se propone esta decisión": rationale,
            "Justificación de la programación": program_reason,
        })

    return pd.DataFrame(rows)


# =============================================================================
# 18. KPI DEL DSS
# =============================================================================

def calculate_kpis(programa: pd.DataFrame, trace: pd.DataFrame, interventions: pd.DataFrame) -> dict:
    latest = trace.sort_values("Fecha_hora").groupby("ID_tramo").tail(1).copy()
    total_tramos = int(latest["ID_tramo"].nunique())
    work_required = int(interventions["INGRESA_MILP"].eq("SI").sum()) if not interventions.empty else 0

    if programa is None or programa.empty:
        programmed = 0
        pending = work_required
        within_rvo = 0
        total_portfolio_budget = 0.0
        programmed_budget = 0.0
        impact = 0.0
        economic_total = 0.0
        hh_total = 0.0
        intervention_hours = 0.0
    else:
        programmed_mask = programa["ESTADO_PROGRAMACION"].eq("PROGRAMADO")
        programmed = int(programmed_mask.sum())
        pending = int(programa["ESTADO_PROGRAMACION"].eq("NO PROGRAMADO").sum())
        within_rvo = int((programmed_mask & programa["CUMPLIMIENTO_RVO"].eq("DENTRO DE PLAZO RVO")).sum())
        total_portfolio_budget = float(programa["PRESUPUESTO_MANTENIMIENTO_S"].sum())
        programmed_budget = float(programa.loc[programmed_mask, "PRESUPUESTO_MANTENIMIENTO_S"].sum())
        impact = float(programa["IMPACTO_OPERATIVO_S"].sum())
        economic_total = float(programa["COSTO_ECONOMICO_TOTAL_S"].sum())
        hh_total = float(programa["HH_TOTAL"].sum())
        intervention_hours = float(programa["HORAS_INTERVENCION"].sum())

    # KPI de utilización planificada de recursos. Se calcula solo con trabajos
    # programados y representa el promedio de utilización de las capacidades
    # semanales del modelo en las semanas efectivamente ocupadas. No se presenta
    # como utilización REAL; esta última requiere datos de ejecución del CMMS.
    resource_utilization_planned = 0.0
    if programa is not None and not programa.empty:
        scheduled = programa[programa["ESTADO_PROGRAMACION"].eq("PROGRAMADO")].copy()
        if not scheduled.empty:
            active_weeks = max(1, int(scheduled["SEMANA_PROGRAMADA"].nunique()))
            ratios = []
            for rec, cap in CAPACIDAD_SEMANAL.items():
                ratios.append(float(scheduled[rec].sum()) / float(cap * active_weeks))
            ratios.append(float(scheduled["HORAS_UT"].sum()) / float(CAP_UT_H_SEMANA * active_weeks))
            ratios.append(float(scheduled["HORAS_EQUIPO_SOLDADURA"].sum()) / float(CAP_SOLDADURA_H_SEMANA * active_weeks))
            resource_utilization_planned = 100.0 * float(np.mean(ratios)) if ratios else 0.0

    return {
        "Tramos evaluados": total_tramos,
        "Tramos alto/crítico": int(latest["NIVEL_RIESGO"].isin(["ALTO", "CRITICO"]).sum()),
        "Trabajos que requieren programación": work_required,
        "Trabajos programados": programmed,
        "Trabajos pendientes de programación": pending,
        "Cumplimiento de programación (%)": round(100 * programmed / work_required, 1) if work_required else 100.0,
        "Cumplimiento del plazo RVO (%)": round(100 * within_rvo / programmed, 1) if programmed else 0.0,
        "PoF posterior promedio": round(float(latest["PoF_posterior"].mean()), 4) if total_tramos else 0.0,
        "Riesgo DRBI promedio": round(float(latest["Riesgo_DRBI"].mean()), 4) if total_tramos else 0.0,
        "Riesgo DRBI máximo": round(float(latest["Riesgo_DRBI"].max()), 4) if total_tramos else 0.0,
        "Presupuesto estimado de la cartera (S/)": round(total_portfolio_budget, 0),
        "Presupuesto ya programado (S/)": round(programmed_budget, 0),
        "Impacto operativo estimado (S/)": round(impact, 0),
        "Costo económico estimado total (S/)": round(economic_total, 0),
        "Horas-hombre estimadas": round(hh_total, 0),
        "Horas de intervención estimadas": round(intervention_hours, 0),
        "Utilización planificada de recursos (%)": round(resource_utilization_planned, 1),
    }


KPI_EXPLICACION = {
    "Tramos evaluados": "Cantidad de tramos del tipo de tubería seleccionado que llegaron a la evaluación DRBI.",
    "Tramos alto/crítico": "Tramos cuyo riesgo actual se clasifica como ALTO o CRÍTICO.",
    "Trabajos que requieren programación": "Intervenciones A2–A6 que pasan al módulo de programación MILP.",
    "Trabajos programados": "Intervenciones a las que el MILP asignó una semana factible.",
    "Trabajos pendientes de programación": "Intervenciones que continúan en la cartera sin semana asignada.",
    "Cumplimiento de programación (%)": "Porcentaje de trabajos requeridos que recibieron una semana en el horizonte.",
    "Cumplimiento del plazo RVO (%)": "Porcentaje de trabajos programados cuya semana no supera el plazo RVO.",
    "PoF posterior promedio": "Promedio de la probabilidad de falla después de incorporar la evidencia IOW.",
    "Riesgo DRBI promedio": "Promedio del índice PoF posterior × CoF en el estado más reciente de cada tramo.",
    "Riesgo DRBI máximo": "Mayor valor actual de riesgo dinámico entre los tramos evaluados.",
    "Presupuesto estimado de la cartera (S/)": "Costo de mantenimiento estimado de todos los trabajos que requieren programación, estén o no programados.",
    "Presupuesto ya programado (S/)": "Costo de mantenimiento correspondiente solo a los trabajos con semana asignada.",
    "Impacto operativo estimado (S/)": "Estimación del costo asociado a la indisponibilidad de la cartera de trabajos.",
    "Costo económico estimado total (S/)": "Presupuesto de mantenimiento más impacto operativo estimado.",
    "Horas-hombre estimadas": "Suma de horas-hombre requeridas por especialidad para la cartera.",
    "Horas de intervención estimadas": "Suma de duraciones estimadas de intervención de la cartera.",
    "Utilización planificada de recursos (%)": "Promedio de uso de capacidades semanales del modelo en las semanas programadas; no equivale a utilización real ejecutada.",
}



# =============================================================================
# 18B. KPI INDIVIDUALES POR TRAMO DE TUBERÍA
# =============================================================================

KPI_TRAMO_EXPLICACION = {
    "PoF actualizada": "Probabilidad de falla del tramo después de incorporar la evidencia IOW más reciente.",
    "Variación de PoF": "Cambio entre la PoF inicial y la PoF actualizada para el tramo seleccionado.",
    "Riesgo DRBI": "Riesgo dinámico del tramo, calculado como PoF actualizada × CoF.",
    "Nivel de riesgo": "Clasificación actual del tramo según el valor DRBI: Bajo, Medio, Alto o Crítico.",
    "IOWs desviadas": "Cantidad de variables IOW que están fuera de sus límites en la última observación.",
    "Evidencia Bayes": "Indicador E(t) que combina magnitud, persistencia y frecuencia de las desviaciones IOW.",
    "Velocidad de corrosión (mpy)": "Velocidad de corrosión observada en la última evidencia disponible del tramo.",
    "Vida remanente (años)": "Vida remanente estimada del tramo según la información cargada al DSS.",
    "Acción recomendada": "Acción prescriptiva seleccionada por el MDP-RVO para este tramo.",
    "Prioridad": "Prioridad de intervención asignada por el modelo.",
    "Fecha límite RVO": "Última fecha permitida por el plazo RVO calculado para este tramo.",
    "Estado de programación": "Indica si el MILP asignó una semana/fecha factible al trabajo de este tramo.",
    "Fecha programada": "Fecha propuesta por el MILP respetando lead time, RVO, ventana y recursos.",
    "Presupuesto de mantenimiento (S/)": "Costo directo e indirecto estimado de la intervención del tramo.",
    "Impacto operativo estimado (S/)": "Costo estimado asociado a la indisponibilidad de este tramo durante la intervención.",
    "Costo económico total estimado (S/)": "Presupuesto de mantenimiento más impacto operativo estimado para el tramo.",
    "Horas-hombre estimadas": "Horas-hombre totales requeridas por el trabajo de este tramo.",
    "Horas de intervención estimadas": "Duración estimada de la intervención del tramo.",
    "Utilización planificada de recursos (%)": "Promedio del uso de capacidades semanales del modelo para la intervención de este tramo.",
}


def calculate_kpis_by_tramo(result: dict) -> pd.DataFrame:
    """Construye KPI independientes para cada ID de tubería/tramo.

    Esta tabla evita mezclar indicadores de tuberías distintas. Cada fila es un
    tramo y se calcula con su última evidencia, su decisión MDP-RVO y, cuando
    corresponde, su programación MILP.
    """
    latest2 = _latest(result["capa2"])
    latest3 = _latest(result["capa3"])
    latest4 = _latest(result["capa4"])
    interventions = result.get("intervenciones", pd.DataFrame())
    program = result.get("programa", pd.DataFrame())

    rows = []
    for tid in sorted(latest4["ID_tramo"].astype(str).unique()):
        r2 = latest2[latest2["ID_tramo"].astype(str).eq(tid)].iloc[-1]
        r3 = latest3[latest3["ID_tramo"].astype(str).eq(tid)].iloc[-1]
        r4 = latest4[latest4["ID_tramo"].astype(str).eq(tid)].iloc[-1]

        r5s = interventions[interventions["ID_tramo"].astype(str).eq(tid)] if not interventions.empty else interventions
        r6s = program[program["ID_tramo"].astype(str).eq(tid)] if not program.empty else program
        r5 = r5s.iloc[0] if len(r5s) else None
        r6 = r6s.iloc[0] if len(r6s) else None

        # Utilización planificada del trabajo individual respecto de una semana.
        util = 0.0
        if r6 is not None and str(r6.get("ESTADO_PROGRAMACION", "")) == "PROGRAMADO":
            ratios = []
            for rec, cap in CAPACIDAD_SEMANAL.items():
                ratios.append(float(r6.get(rec, 0.0) or 0.0) / float(cap))
            ratios.append(float(r6.get("HORAS_UT", 0.0) or 0.0) / float(CAP_UT_H_SEMANA))
            ratios.append(float(r6.get("HORAS_EQUIPO_SOLDADURA", 0.0) or 0.0) / float(CAP_SOLDADURA_H_SEMANA))
            util = 100.0 * float(np.mean(ratios)) if ratios else 0.0

        rows.append({
            "Tramo": tid,
            "Tipo de tubería": r4.get("TIPO_TUBERIA", result.get("tipo", "")),
            "Servicio / fluido": r4.get("Servicio_fluido", ""),
            "Fecha de corte": result.get("fecha_corte_analisis", pd.NaT),
            "PoF inicial": round(float(r3.get("PoF_prior", 0.0)), 4),
            "PoF actualizada": round(float(r3.get("PoF_posterior", 0.0)), 4),
            "Variación de PoF": round(float(r3.get("Delta_PoF", 0.0)), 4),
            "CoF": round(float(r4.get("CoF_valor", 0.0)), 2),
            "Riesgo DRBI": round(float(r4.get("Riesgo_DRBI", 0.0)), 4),
            "Nivel de riesgo": r4.get("NIVEL_RIESGO", ""),
            "IOWs desviadas": int(r2.get("N_IOW_DESVIADAS", 0)),
            "Magnitud M(t)": round(float(r2.get("M_t", 0.0)), 4),
            "Persistencia D(t)": round(float(r2.get("D_t", 0.0)), 4),
            "Frecuencia F(t)": round(float(r2.get("F_t", 0.0)), 4),
            "Evidencia Bayes": round(float(r2.get("E_t", 0.0)), 4),
            "Espesor actual (mm)": round(float(r4.get("Espesor_actual_mm", 0.0)), 3),
            "Velocidad de corrosión (mpy)": round(float(r4.get("Velocidad_corrosion_mpy", 0.0)), 3),
            "Vida remanente (años)": round(float(r4.get("Vida_remanente_anios", 0.0)), 2),
            "Acción": r5.get("ACCION_MDP_ID", "A1") if r5 is not None else "A1",
            "Acción recomendada": r5.get("ESTRATEGIA_MANTENIMIENTO", "Monitoreo") if r5 is not None else "Monitoreo",
            "Prioridad": r5.get("PRIORIDAD_MTTO", "Seguimiento") if r5 is not None else "Seguimiento",
            "Fecha límite RVO": r5.get("FECHA_LIMITE_PLANIFICACION", pd.NaT) if r5 is not None else pd.NaT,
            "Estado RVO": r5.get("ESTADO_RVO", "MONITOREO") if r5 is not None else "MONITOREO",
            "Estado de programación": r6.get("ESTADO_PROGRAMACION", "NO REQUIERE PROGRAMACIÓN") if r6 is not None else "NO REQUIERE PROGRAMACIÓN",
            "Semana programada": r6.get("SEMANA_PROGRAMADA", np.nan) if r6 is not None else np.nan,
            "Fecha programada": r6.get("FECHA_PROGRAMADA", pd.NaT) if r6 is not None else pd.NaT,
            "Cumplimiento RVO": r6.get("CUMPLIMIENTO_RVO", "NO APLICA") if r6 is not None else "NO APLICA",
            "Presupuesto de mantenimiento (S/)": round(float(r6.get("PRESUPUESTO_MANTENIMIENTO_S", 0.0) or 0.0), 2) if r6 is not None else 0.0,
            "Impacto operativo estimado (S/)": round(float(r6.get("IMPACTO_OPERATIVO_S", 0.0) or 0.0), 2) if r6 is not None else 0.0,
            "Costo económico total estimado (S/)": round(float(r6.get("COSTO_ECONOMICO_TOTAL_S", 0.0) or 0.0), 2) if r6 is not None else 0.0,
            "Horas-hombre estimadas": round(float(r6.get("HH_TOTAL", 0.0) or 0.0), 1) if r6 is not None else 0.0,
            "Horas de intervención estimadas": round(float(r6.get("HORAS_INTERVENCION", 0.0) or 0.0), 1) if r6 is not None else 0.0,
            "Utilización planificada de recursos (%)": round(util, 1),
        })

    return pd.DataFrame(rows)


def kpis_tramo_dict(result: dict, tramo: str) -> dict:
    """Devuelve los KPI principales de un tramo como diccionario ordenado."""
    table = result.get("kpis_por_tramo")
    if table is None or not isinstance(table, pd.DataFrame):
        table = calculate_kpis_by_tramo(result)
    row = table[table["Tramo"].astype(str).eq(str(tramo))]
    if row.empty:
        raise KeyError(f"No se encontró el tramo {tramo} en el resultado actual.")
    r = row.iloc[0]
    keys = [
        "PoF actualizada", "Variación de PoF", "Riesgo DRBI", "Nivel de riesgo",
        "IOWs desviadas", "Evidencia Bayes", "Velocidad de corrosión (mpy)",
        "Vida remanente (años)", "Acción recomendada", "Prioridad",
        "Fecha límite RVO", "Estado de programación", "Fecha programada",
        "Presupuesto de mantenimiento (S/)", "Impacto operativo estimado (S/)",
        "Costo económico total estimado (S/)", "Horas-hombre estimadas",
        "Horas de intervención estimadas", "Utilización planificada de recursos (%)",
    ]
    return {k: r.get(k) for k in keys}


def history_tramo(result: dict, tramo: str) -> pd.DataFrame:
    """Serie histórica compacta usada por el dashboard individual del tramo."""
    cols = [
        "Fecha_hora", "Periodo_semana", "ID_tramo", "PoF_prior", "PoF_posterior",
        "Delta_PoF", "Riesgo_DRBI", "NIVEL_RIESGO", "N_IOW_DESVIADAS",
        "M_t", "D_t", "F_t", "E_t",
    ]
    d = result["capa4"].copy()
    d = d[d["ID_tramo"].astype(str).eq(str(tramo))].sort_values("Fecha_hora")
    return d[cols].copy()


# =============================================================================
# 19. EJECUCIÓN INTEGRADA DE LAS 7 CAPAS
# =============================================================================

def run_all(
    df: pd.DataFrame,
    tipo: str,
    windows: Optional[pd.DataFrame] = None,
    iow_params: Optional[Mapping[str, Sequence[float]]] = None,
    presupuesto_mensual: float = PRESUPUESTO_MENSUAL_DEFAULT,
    semana_revision: Optional[int] = None,
    horizonte_semanas: int = HORIZONTE_SEMANAS_DEFAULT,
    auto_generate_windows: bool = True,
) -> dict:
    l1, quality = layer1_etl(df, tipo)
    l2 = layer2_iows(l1, iow_params)
    l3 = layer3_bayes(l2)
    l4 = layer4_drbi(l3)
    # La fecha de corte es la evidencia más reciente del conjunto analizado.
    # Esta fecha inicia el nuevo ciclo de planificación y evita fechas históricas
    # anteriores al estado que acaba de evaluar el DSS.
    fecha_corte_analisis = pd.Timestamp(l1["Fecha_hora"].max()).normalize()
    interventions, policy = layer5_mdp_rvo(
        l4,
        windows,
        auto_generate_windows=auto_generate_windows,
        fecha_corte_analisis=fecha_corte_analisis,
    )
    program = layer6_milp(
        interventions,
        horizonte_semanas=horizonte_semanas,
        presupuesto_mensual=presupuesto_mensual,
        fecha_inicio=fecha_corte_analisis,
    )
    rolling = rolling_horizon(program, l4, semana_revision=semana_revision)
    kpi = calculate_kpis(program, l4, interventions)

    result = {
        "version": VERSION_DSS,
        "tipo": tipo,
        "fecha_corte_analisis": fecha_corte_analisis,
        "capa1": l1,
        "capa2": l2,
        "capa3": l3,
        "capa4": l4,
        "intervenciones": interventions,
        "politica_mdp": policy,
        "programa": program,
        "rolling": rolling,
        "kpis": kpi,
        "politicas": maintenance_policy(tipo),
        "politica_programacion": scheduling_policy(tipo),
        "calidad": quality,
        "glosario": GLOSARIO.copy(),
    }
    result["politica_detallada_tramo"] = policy_decision_detail(result, tipo)
    result["kpis_por_tramo"] = calculate_kpis_by_tramo(result)
    return result


# =============================================================================
# 20. TABLAS DE PRESENTACIÓN (NOMBRES COMPRENSIBLES)
# =============================================================================

def _latest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_values("Fecha_hora").groupby("ID_tramo").tail(1).copy()


def table_layer1(result: dict) -> pd.DataFrame:
    cols = [
        "Fecha_hora", "Periodo_semana", "ID_tramo", "TIPO_TUBERIA", "Servicio_fluido",
        "Espesor_actual_mm", "Velocidad_corrosion_mpy", "Vida_remanente_anios",
        "Temperatura_C", "Presion_bar_g",
    ]
    out = result["capa1"][cols].copy()
    return out.rename(columns={
        "Fecha_hora": "Fecha",
        "Periodo_semana": "Semana",
        "ID_tramo": "Tramo",
        "TIPO_TUBERIA": "Tipo de tubería",
        "Servicio_fluido": "Servicio / fluido",
        "Espesor_actual_mm": "Espesor actual (mm)",
        "Velocidad_corrosion_mpy": "Velocidad de corrosión (mpy)",
        "Vida_remanente_anios": "Vida remanente (años)",
        "Temperatura_C": "Temperatura (°C)",
        "Presion_bar_g": "Presión (bar g)",
    })


def table_layer2(result: dict) -> pd.DataFrame:
    d = result["capa2"].copy()
    cols = ["Fecha_hora", "Periodo_semana", "ID_tramo", "N_IOW_DESVIADAS", "INDICE_IOW", "NIVEL_IOW", "M_t", "D_t", "F_t", "E_t", "NIVEL_EVIDENCIA"]
    return d[cols].rename(columns={
        "Fecha_hora": "Fecha",
        "Periodo_semana": "Semana",
        "ID_tramo": "Tramo",
        "N_IOW_DESVIADAS": "Variables IOW fuera de límite",
        "INDICE_IOW": "Índice de desviación IOW (0-1)",
        "NIVEL_IOW": "Nivel de desviación",
        "M_t": "Magnitud promedio de desviación (0-1)",
        "D_t": "Persistencia de la desviación (0-1)",
        "F_t": "Frecuencia de desviación (0-1)",
        "E_t": "Evidencia combinada para Bayes (0-1)",
        "NIVEL_EVIDENCIA": "Nivel de evidencia",
    })


def table_layer3(result: dict) -> pd.DataFrame:
    d = result["capa3"].copy()
    cols = ["Fecha_hora", "Periodo_semana", "ID_tramo", "E_t", "PoF_prior", "P_E_dado_F", "P_E_dado_noF", "PoF_posterior", "Delta_PoF", "NIVEL_PoF"]
    return d[cols].rename(columns={
        "Fecha_hora": "Fecha",
        "Periodo_semana": "Semana",
        "ID_tramo": "Tramo",
        "E_t": "Evidencia IOW combinada",
        "PoF_prior": "PoF inicial",
        "P_E_dado_F": "Verosimilitud con falla P(E|F)",
        "P_E_dado_noF": "Verosimilitud sin falla P(E|¬F)",
        "PoF_posterior": "PoF actualizada",
        "Delta_PoF": "Cambio de PoF",
        "NIVEL_PoF": "Nivel de PoF",
    })


def table_layer4(result: dict) -> pd.DataFrame:
    d = result["capa4"].copy()
    cols = ["Fecha_hora", "Periodo_semana", "ID_tramo", "Servicio_fluido", "PoF_posterior", "CoF_valor", "Riesgo_DRBI", "Riesgo_normalizado", "NIVEL_RIESGO"]
    return d[cols].rename(columns={
        "Fecha_hora": "Fecha",
        "Periodo_semana": "Semana",
        "ID_tramo": "Tramo",
        "Servicio_fluido": "Servicio / fluido",
        "PoF_posterior": "PoF actualizada",
        "CoF_valor": "Consecuencia de falla (CoF)",
        "Riesgo_DRBI": "Riesgo dinámico DRBI",
        "Riesgo_normalizado": "Riesgo normalizado (0-1)",
        "NIVEL_RIESGO": "Nivel de riesgo",
    })


def table_layer5(result: dict) -> pd.DataFrame:
    d = result["intervenciones"].copy()
    cols = [
        "ID_SOLICITUD", "ID_tramo", "Servicio_fluido", "NIVEL_RIESGO", "Riesgo_DRBI",
        "ACCION_MDP_ID", "ESTRATEGIA_MANTENIMIENTO", "PRIORIDAD_MTTO", "URGENCIA_RVO",
        "FECHA_CORTE_ANALISIS", "FECHA_DETECCION", "FECHA_ULTIMA_EVIDENCIA_TRAMO", "ANTIGUEDAD_DATOS_DIAS",
        "LEAD_TIME_DIAS", "FECHA_DISPONIBLE_DESDE", "PLAZO_RVO_DIAS", "FECHA_LIMITE_PLANIFICACION", "ESTADO_RVO",
        "VENTANA_ID", "VENTANA_INICIO_REAL", "ORIGEN_VENTANA", "ALERTA_VIDA_REMANENTE",
    ]
    return d[cols].rename(columns={
        "ID_SOLICITUD": "Solicitud",
        "ID_tramo": "Tramo",
        "Servicio_fluido": "Servicio / fluido",
        "NIVEL_RIESGO": "Nivel de riesgo",
        "Riesgo_DRBI": "Riesgo DRBI",
        "ACCION_MDP_ID": "Código de acción",
        "ESTRATEGIA_MANTENIMIENTO": "Acción de mantenimiento recomendada",
        "PRIORIDAD_MTTO": "Prioridad",
        "URGENCIA_RVO": "Urgencia",
        "FECHA_CORTE_ANALISIS": "Fecha de corte del análisis",
        "FECHA_DETECCION": "Primera detección histórica del nivel máximo",
        "FECHA_ULTIMA_EVIDENCIA_TRAMO": "Última evidencia del tramo",
        "ANTIGUEDAD_DATOS_DIAS": "Antigüedad de datos (días)",
        "LEAD_TIME_DIAS": "Tiempo mínimo de preparación (días)",
        "FECHA_DISPONIBLE_DESDE": "Primera fecha técnicamente disponible",
        "PLAZO_RVO_DIAS": "Plazo máximo RVO (días)",
        "FECHA_LIMITE_PLANIFICACION": "Fecha límite RVO",
        "ESTADO_RVO": "Estado de ventana / RVO",
        "VENTANA_ID": "Ventana seleccionada",
        "VENTANA_INICIO_REAL": "Inicio de ventana",
        "ORIGEN_VENTANA": "Origen / fundamento de la ventana",
        "ALERTA_VIDA_REMANENTE": "Validación con vida remanente",
    })


def table_layer6(result: dict) -> pd.DataFrame:
    d = result["programa"].copy()
    if d.empty:
        return d
    cols = [
        "ID_SOLICITUD", "ID_tramo", "NIVEL_RIESGO", "ESTRATEGIA_MANTENIMIENTO", "PRIORIDAD_MTTO",
        "FECHA_CORTE_ANALISIS", "LEAD_TIME_DIAS_MILP", "FECHA_MAS_TEMPRANA", "PLAZO_RVO_DIAS", "FECHA_LIMITE_PLANIFICACION",
        "VENTANA_INICIO_REAL", "VENTANA_FIN_REAL", "ORIGEN_VENTANA",
        "ESTADO_PROGRAMACION", "SEMANA_PROGRAMADA", "FECHA_PROGRAMADA", "CUMPLIMIENTO_RVO",
        "MOTIVO_NO_PROGRAMACION", "HH_TOTAL", "HORAS_INTERVENCION", "PRESUPUESTO_MANTENIMIENTO_S",
        "IMPACTO_OPERATIVO_S", "COSTO_ECONOMICO_TOTAL_S", "SOLVER_MILP",
    ]
    return d[cols].rename(columns={
        "ID_SOLICITUD": "Solicitud",
        "ID_tramo": "Tramo",
        "NIVEL_RIESGO": "Nivel de riesgo",
        "ESTRATEGIA_MANTENIMIENTO": "Acción de mantenimiento",
        "PRIORIDAD_MTTO": "Prioridad",
        "FECHA_CORTE_ANALISIS": "Fecha de corte del análisis",
        "LEAD_TIME_DIAS_MILP": "Tiempo mínimo de preparación (días)",
        "FECHA_MAS_TEMPRANA": "Primera fecha posible",
        "PLAZO_RVO_DIAS": "Plazo máximo RVO (días)",
        "FECHA_LIMITE_PLANIFICACION": "Fecha límite RVO",
        "VENTANA_INICIO_REAL": "Inicio de ventana",
        "VENTANA_FIN_REAL": "Fin de ventana",
        "ORIGEN_VENTANA": "Origen de la ventana",
        "ESTADO_PROGRAMACION": "Estado de programación",
        "SEMANA_PROGRAMADA": "Semana programada",
        "FECHA_PROGRAMADA": "Fecha programada",
        "CUMPLIMIENTO_RVO": "Cumplimiento del plazo RVO",
        "MOTIVO_NO_PROGRAMACION": "Fundamento de la fecha / explicación de programación",
        "HH_TOTAL": "Horas-hombre",
        "HORAS_INTERVENCION": "Horas de intervención",
        "PRESUPUESTO_MANTENIMIENTO_S": "Presupuesto mantenimiento (S/)",
        "IMPACTO_OPERATIVO_S": "Impacto operativo estimado (S/)",
        "COSTO_ECONOMICO_TOTAL_S": "Costo económico total (S/)",
        "SOLVER_MILP": "Motor de optimización",
    })


def table_layer7(result: dict) -> pd.DataFrame:
    d = result["rolling"].copy()
    if d.empty:
        return d
    cols = [
        "ID_SOLICITUD", "ID_tramo", "NIVEL_RIESGO", "NIVEL_RIESGO_ACTUAL", "RIESGO_ANTERIOR",
        "RIESGO_ACTUAL", "DELTA_RIESGO", "VARIACION_RIESGO_PCT", "ESTADO_HORIZONTE",
        "DECISION_ROLLING_HORIZON", "ACCION_PLANIFICACION",
    ]
    return d[cols].rename(columns={
        "ID_SOLICITUD": "Solicitud",
        "ID_tramo": "Tramo",
        "NIVEL_RIESGO": "Riesgo considerado al programar",
        "NIVEL_RIESGO_ACTUAL": "Riesgo actual",
        "RIESGO_ANTERIOR": "Índice de riesgo anterior",
        "RIESGO_ACTUAL": "Índice de riesgo actual",
        "DELTA_RIESGO": "Cambio del riesgo",
        "VARIACION_RIESGO_PCT": "Variación del riesgo (%)",
        "ESTADO_HORIZONTE": "Estado en el horizonte",
        "DECISION_ROLLING_HORIZON": "Decisión de reprogramación",
        "ACCION_PLANIFICACION": "Acción de planificación",
    })


def temporal_basis_table(result: dict) -> pd.DataFrame:
    """Explica de forma auditable cómo se derivan las fechas del programa."""
    fecha_corte = pd.Timestamp(result.get("fecha_corte_analisis")).normalize()
    return pd.DataFrame(
        [
            ["Fecha de corte del análisis", fecha_corte, "Última fecha disponible en los datos seleccionados; inicia el nuevo ciclo de planificación."],
            ["Primera fecha posible", "Fecha de corte + lead time de la acción", "Evita programar antes del tiempo mínimo de preparación técnica/logística."],
            ["Fecha límite RVO", "Fecha de corte + plazo RVO del nivel de riesgo", "Representa el plazo máximo seguro utilizado por MDP-RVO para el ciclo actual."],
            ["Ventana operativa", "Ventana cargada o ventana provisional", "Si el usuario carga ventanas se usa la primera compatible; si no, el DSS propone una ventana provisional que debe confirmarse."],
            ["Fecha programada", "max(inicio del periodo MILP, primera fecha posible, inicio de ventana)", "La fecha exacta nunca es anterior al lead time ni a la ventana seleccionada."],
            ["Reprogramación", "Rolling Horizon", "La fecha puede cambiar cuando cambia DRBI, RVO, ventanas, recursos o presupuesto."],
        ],
        columns=["Elemento temporal", "Base de cálculo", "Justificación"],
    )


def table_layer7_kpis(result: dict) -> pd.DataFrame:
    """KPI de la Capa 7 que sí pueden calcularse con la salida actual del DSS.

    Se distingue explícitamente entre KPI de planificación y KPI posteriores a la
    ejecución. Esto evita presentar como resultado real aquello que todavía
    requiere órdenes de trabajo ejecutadas, paradas reales y datos operacionales.
    """
    k = result["kpis"]
    rows = [
        ["Tramos evaluados", k.get("Tramos evaluados", 0), "tramos", "Conteo de tramos que llegaron a DRBI", "Cobertura del análisis", "CALCULADO"],
        ["Tramos alto/crítico", k.get("Tramos alto/crítico", 0), "tramos", "Conteo de tramos en ALTO o CRÍTICO", "Reducir exposición prioritaria", "CALCULADO"],
        ["Cumplimiento de programación", k.get("Cumplimiento de programación (%)", 0), "%", "Trabajos programados / trabajos requeridos × 100", "Maximizar sin violar restricciones", "CALCULADO"],
        ["Cumplimiento del plazo RVO", k.get("Cumplimiento del plazo RVO (%)", 0), "%", "Trabajos programados dentro del RVO / programados × 100", "Objetivo: 100 %", "CALCULADO"],
        ["PoF posterior promedio", k.get("PoF posterior promedio", 0), "probabilidad", "Promedio de PoF posterior del estado más reciente", "Seguimiento; menor es mejor", "CALCULADO"],
        ["Riesgo DRBI promedio", k.get("Riesgo DRBI promedio", 0), "índice", "Promedio PoF posterior × CoF", "Seguimiento; menor es mejor", "CALCULADO"],
        ["Riesgo DRBI máximo", k.get("Riesgo DRBI máximo", 0), "índice", "Máximo riesgo actual entre tramos", "Mantener dentro del criterio aprobado", "CALCULADO"],
        ["Costo económico estimado total", k.get("Costo económico estimado total (S/)", 0), "S/", "Mantenimiento + impacto operativo estimado", "Optimizar sin deteriorar seguridad", "CALCULADO"],
        ["Horas-hombre estimadas", k.get("Horas-hombre estimadas", 0), "HH", "Suma de HH requeridas por la cartera", "Compatible con capacidad", "CALCULADO"],
        ["Horas de intervención estimadas", k.get("Horas de intervención estimadas", 0), "h", "Suma de duraciones estimadas", "Compatible con ventanas", "CALCULADO"],
        ["Utilización planificada de recursos", k.get("Utilización planificada de recursos (%)", 0), "%", "Promedio de uso/capacidad semanal en semanas ocupadas", "Uso factible y sin conflictos", "CALCULADO"],
    ]
    return pd.DataFrame(rows, columns=["KPI", "Resultado", "Unidad", "Fórmula / criterio", "Criterio de evaluación", "Estado"] )


def table_layer7_pending_kpis(result: dict) -> pd.DataFrame:
    """KPI de la tesis que requieren evidencia posterior a la ejecución."""
    rows = [
        ["Número de paradas no programadas", "Pendiente", "eventos/periodo", "Conteo de eventos atribuibles a corrosión", "Registros de operación e incidentes", "Requiere línea base y periodo posterior comparable"],
        ["Horas de indisponibilidad", "Pendiente", "h/periodo", "Σ horas de parada no programada", "Registros de operación", "Requiere horas reales posteriores a la implementación"],
        ["Disponibilidad operacional", "Pendiente", "%", "T_operativo /(T_operativo + T_parada) × 100", "DCS / registros de operación", "Requiere tiempos operativos y de parada reales"],
        ["Riesgo residual posterior", "Pendiente", "índice", "PoF posterior × CoF después de inspección/intervención", "DRBI / inspección posterior", "Requiere nueva evidencia después de ejecutar la acción"],
        ["Cumplimiento real del mantenimiento", "Pendiente", "%", "OT ejecutadas dentro del plazo / OT programadas × 100", "CMMS / órdenes ejecutadas", "Requiere fecha real de ejecución"],
        ["Costo real total", "Pendiente", "S/ por periodo", "Costos directos + pérdida de producción + otros costos sin doble conteo", "Mantenimiento / Finanzas / Operaciones", "Requiere costos reales cerrados"],
        ["Utilización real de recursos", "Pendiente", "%", "Capacidad utilizada / capacidad disponible × 100", "Programación / CMMS", "Requiere HH y disponibilidad real de recursos"],
    ]
    return pd.DataFrame(rows, columns=["KPI", "Resultado", "Unidad", "Fórmula / criterio", "Fuente requerida", "Condición para validarlo"])


def current_status_table(result: dict) -> pd.DataFrame:
    """Resumen más reciente por tramo para dashboard y Excel."""
    latest = _latest(result["capa4"])
    interventions = result["intervenciones"]
    cols_i = [
        "ID_tramo", "ACCION_MDP_ID", "ESTRATEGIA_MANTENIMIENTO", "PRIORIDAD_MTTO",
        "FECHA_LIMITE_PLANIFICACION", "ESTADO_RVO",
    ]
    out = latest.merge(interventions[cols_i], on="ID_tramo", how="left")
    return out[[
        "ID_tramo", "Servicio_fluido", "PoF_posterior", "CoF_valor", "Riesgo_DRBI", "NIVEL_RIESGO",
        "ACCION_MDP_ID", "ESTRATEGIA_MANTENIMIENTO", "PRIORIDAD_MTTO", "FECHA_LIMITE_PLANIFICACION", "ESTADO_RVO",
    ]].rename(columns={
        "ID_tramo": "Tramo",
        "Servicio_fluido": "Servicio / fluido",
        "PoF_posterior": "PoF actualizada",
        "CoF_valor": "CoF",
        "Riesgo_DRBI": "Riesgo DRBI",
        "NIVEL_RIESGO": "Nivel de riesgo",
        "ACCION_MDP_ID": "Acción",
        "ESTRATEGIA_MANTENIMIENTO": "Acción recomendada",
        "PRIORIDAD_MTTO": "Prioridad",
        "FECHA_LIMITE_PLANIFICACION": "Fecha límite RVO",
        "ESTADO_RVO": "Estado RVO",
    })


# =============================================================================
# 21. VALIDACIONES AUTOMÁTICAS DEL MOTOR
# =============================================================================

def validate_result(result: dict) -> pd.DataFrame:
    checks: List[dict] = []

    def add(name: str, condition: bool, detail: str) -> None:
        checks.append({"Validación": name, "Estado": "OK" if condition else "REVISAR", "Detalle": detail})

    l2, l3, l4 = result["capa2"], result["capa3"], result["capa4"]
    add("Pesos de evidencia IOW", np.isclose(PESO_MAGNITUD + PESO_DURACION + PESO_FRECUENCIA, 1.0), "0.45 + 0.30 + 0.25 = 1.00")
    add("Evidencia IOW en rango", bool(l2["E_t"].between(0, 1).all()), "E_t debe permanecer entre 0 y 1.")
    add("PoF posterior en rango", bool(l3["PoF_posterior"].between(0, 1).all()), "PoF posterior debe permanecer entre 0 y 1.")
    add("Riesgo DRBI no negativo", bool((l4["Riesgo_DRBI"] >= 0).all()), "Riesgo = PoF posterior × CoF.")
    add("Matrices MDP", True, "Las matrices se validan antes de ejecutar value iteration.")

    program = result["programa"]
    if not program.empty:
        add("Una fila por trabajo MILP", bool(program["ID_SOLICITUD"].is_unique), "Cada solicitud debe aparecer una vez en la salida de programación.")
        add("Estado de programación válido", bool(program["ESTADO_PROGRAMACION"].isin(["PROGRAMADO", "NO PROGRAMADO"]).all()), "Estados permitidos: PROGRAMADO / NO PROGRAMADO.")
    else:
        add("Programa MILP", True, "No existen acciones A2–A6 para programar en este escenario.")

    return pd.DataFrame(checks)


def self_test() -> pd.DataFrame:
    """Ejecuta pruebas de regresión funcional sobre los tres tipos de tubería."""
    rows = []
    data = demo_dataset()
    for tipo in TIPOS_TUBERIA:
        result = run_all(data, tipo, windows=None, auto_generate_windows=True)
        validation = validate_result(result)
        ok = bool(validation["Estado"].eq("OK").all())
        rows.append(
            {
                "Tipo de tubería": tipo,
                "Registros": len(result["capa1"]),
                "Tramos": result["capa1"]["ID_tramo"].nunique(),
                "Trabajos MILP": len(result["programa"]),
                "Programados": int(result["programa"].get("ESTADO_PROGRAMACION", pd.Series(dtype=str)).eq("PROGRAMADO").sum()),
                "Estado": "OK" if ok else "REVISAR",
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 22. EXPORTACIÓN EXCEL PARA USO EN GOOGLE COLAB
# =============================================================================
# Se usa XlsxWriter porque está disponible vía pip en Colab y permite controlar
# claramente títulos, anchos, tablas, formatos, gráficos y paneles congelados.
# El reporte mantiene solo tres hojas útiles. Las políticas permanecen en la interfaz, no en Excel.


def export_excel(result: dict, tipo: str, out_path: Optional[str] = None) -> str:
    """Exporta el reporte final con un dashboard interactivo por tramo.

    El reporte conserva tres hojas y no exporta políticas. La primera hoja usa
    una lista desplegable para seleccionar el ID de tubería/tramo; los KPI y
    los gráficos cambian de forma independiente para el tramo elegido.
    """
    import xlsxwriter

    if out_path is None:
        out_path = os.path.join(
            tempfile.gettempdir(),
            f"Reporte_DSS_{tipo}_{pd.Timestamp.now():%Y%m%d_%H%M%S}.xlsx",
        )

    per_pipe = result.get("kpis_por_tramo")
    if per_pipe is None or not isinstance(per_pipe, pd.DataFrame) or per_pipe.empty:
        per_pipe = calculate_kpis_by_tramo(result)
    per_pipe = per_pipe.copy().reset_index(drop=True)
    tramos = per_pipe["Tramo"].astype(str).tolist()
    default_tramo = tramos[0] if tramos else ""

    hist = result["capa4"].copy().sort_values(["Periodo_semana", "ID_tramo"])
    program = table_layer6(result).copy()
    rolling = table_layer7(result).copy()
    current = current_status_table(result).copy()

    # Matrices históricas para gráficos dinámicos por selector.
    weeks = sorted(hist["Periodo_semana"].dropna().astype(int).unique().tolist())
    metric_pivots = {}
    for col in ["PoF_prior", "PoF_posterior", "Riesgo_DRBI", "E_t", "N_IOW_DESVIADAS"]:
        p = hist.pivot_table(index="Periodo_semana", columns="ID_tramo", values=col, aggfunc="last")
        p = p.reindex(index=weeks, columns=tramos)
        metric_pivots[col] = p

    wb = xlsxwriter.Workbook(out_path, {"nan_inf_to_errors": True})
    wb.set_properties({
        "title": f"DSS de Integridad y Mantenimiento - {tipo}",
        "subject": NOMBRE_MODELO,
        "author": "DSS de Integridad y Mantenimiento",
        "comments": "Reporte final con KPI y dashboard individual por tramo.",
    })

    NAVY = "#17365D"; NAVY_DARK = "#0B1F3A"; BLUE = "#2E75B6"
    GREEN = "#2E7D32"; AMBER = "#D98E04"; ORANGE = "#F57C00"; RED = "#C62828"
    BORDER = "#D9E2F3"; TEXT = "#17243A"; MUTED = "#667085"; LIGHT = "#F5F8FC"

    title_fmt = wb.add_format({"bold": True, "font_size": 18, "font_color": "white", "bg_color": NAVY_DARK, "align": "left", "valign": "vcenter"})
    subtitle_fmt = wb.add_format({"font_size": 10, "font_color": MUTED, "italic": True, "align": "left", "valign": "vcenter"})
    section_fmt = wb.add_format({"bold": True, "font_size": 12, "font_color": "white", "bg_color": BLUE, "align": "left", "valign": "vcenter"})
    header_fmt = wb.add_format({"bold": True, "font_color": "white", "bg_color": NAVY, "border": 1, "border_color": BORDER, "align": "center", "valign": "vcenter", "text_wrap": True})
    cell_fmt = wb.add_format({"font_color": TEXT, "border": 1, "border_color": BORDER, "valign": "top", "text_wrap": True})
    card_label = wb.add_format({"bold": True, "font_size": 10, "font_color": MUTED, "bg_color": LIGHT, "align": "center", "valign": "vcenter", "border": 1, "border_color": BORDER})
    card_num = wb.add_format({"bold": True, "font_size": 19, "font_color": NAVY, "bg_color": "white", "align": "center", "valign": "vcenter", "border": 1, "border_color": BORDER, "num_format": "0.0000"})
    card_int = wb.add_format({"bold": True, "font_size": 19, "font_color": NAVY, "bg_color": "white", "align": "center", "valign": "vcenter", "border": 1, "border_color": BORDER, "num_format": "0"})
    card_money = wb.add_format({"bold": True, "font_size": 16, "font_color": NAVY, "bg_color": "white", "align": "center", "valign": "vcenter", "border": 1, "border_color": BORDER, "num_format": "#,##0.00"})
    card_text = wb.add_format({"bold": True, "font_size": 12, "font_color": NAVY, "bg_color": "white", "align": "center", "valign": "vcenter", "border": 1, "border_color": BORDER, "text_wrap": True})
    date_fmt = wb.add_format({"font_color": TEXT, "border": 1, "border_color": BORDER, "num_format": "dd/mm/yyyy", "valign": "top"})
    num_fmt = wb.add_format({"font_color": TEXT, "border": 1, "border_color": BORDER, "num_format": "0.0000", "valign": "top"})
    money_fmt = wb.add_format({"font_color": TEXT, "border": 1, "border_color": BORDER, "num_format": "#,##0.00", "valign": "top"})
    percent_fmt = wb.add_format({"font_color": TEXT, "border": 1, "border_color": BORDER, "num_format": "0.0", "valign": "top"})
    selector_fmt = wb.add_format({"bold": True, "font_size": 12, "font_color": NAVY, "bg_color": "#EAF2FB", "border": 1, "border_color": BLUE, "align": "center", "valign": "vcenter"})
    note_fmt = wb.add_format({"font_color": MUTED, "italic": True, "text_wrap": True, "valign": "top"})

    risk_fmt = {
        "BAJO": wb.add_format({"bg_color": "#E2F0D9", "font_color": "#375623", "bold": True, "align": "center", "border": 1, "border_color": BORDER}),
        "MEDIO": wb.add_format({"bg_color": "#FFF2CC", "font_color": "#7F6000", "bold": True, "align": "center", "border": 1, "border_color": BORDER}),
        "ALTO": wb.add_format({"bg_color": "#FCE4D6", "font_color": "#C65911", "bold": True, "align": "center", "border": 1, "border_color": BORDER}),
        "CRITICO": wb.add_format({"bg_color": "#F8CBAD", "font_color": "#9C0006", "bold": True, "align": "center", "border": 1, "border_color": BORDER}),
    }

    # ------------------------------------------------------------------
    # HOJA 1: Dashboard_KPI
    # ------------------------------------------------------------------
    ws = wb.add_worksheet("Dashboard_KPI")
    ws.hide_gridlines(2)
    ws.set_row(0, 32)
    ws.merge_range("A1:P1", f"CAPA 7 – KPI Y DASHBOARD INDIVIDUAL | TUBERÍAS DE {tipo.upper()}", title_fmt)
    ws.merge_range("A2:P2", f"Seleccione un tramo para ver únicamente sus KPI, tendencias y decisión. Fecha de corte: {pd.Timestamp(result['fecha_corte_analisis']):%d/%m/%Y}.", subtitle_fmt)
    ws.write("A3", "Tubería / tramo seleccionado", header_fmt)
    ws.write("B3", default_tramo, selector_fmt)
    if tramos:
        ws.data_validation("B3", {"validate": "list", "source": "=$AA$2:$AA$%d" % (len(tramos)+1), "input_title": "Seleccione un tramo", "input_message": "El dashboard se actualiza para el tramo seleccionado."})
    ws.merge_range("D3:P3", "Los KPI de esta hoja no mezclan tuberías: cada valor corresponde al ID elegido en B3.", note_fmt)

    # Tabla maestra oculta de KPI por tramo: AA:BH.
    hidden_start_col = 26  # AA
    for j, col in enumerate(per_pipe.columns):
        ws.write(0, hidden_start_col + j, col, header_fmt)
    for i, (_, row) in enumerate(per_pipe.iterrows(), start=1):
        for j, col in enumerate(per_pipe.columns):
            val = row[col]
            if pd.isna(val): val = ""
            if isinstance(val, pd.Timestamp):
                ws.write_datetime(i, hidden_start_col + j, val.to_pydatetime(), date_fmt)
            else:
                ws.write(i, hidden_start_col + j, val)

    # Fórmula INDEX/MATCH reutilizable.
    last_hidden_col = xlsxwriter.utility.xl_col_to_name(hidden_start_col + len(per_pipe.columns) - 1)
    def lookup_formula(field):
        return f'=IFERROR(INDEX($AA$2:${last_hidden_col}${len(tramos)+1},MATCH($B$3,$AA$2:$AA${len(tramos)+1},0),MATCH("{field}",$AA$1:${last_hidden_col}$1,0)),"")'

    field_cache = {}
    if default_tramo:
        rr = per_pipe[per_pipe["Tramo"].astype(str).eq(default_tramo)].iloc[0]
        field_cache = rr.to_dict()

    cards = [
        ("A5:C5", "A6:C8", "PoF actualizada", "PoF actualizada", card_num),
        ("D5:F5", "D6:F8", "Riesgo DRBI", "Riesgo DRBI", card_num),
        ("G5:I5", "G6:I8", "Nivel de riesgo", "Nivel de riesgo", card_text),
        ("J5:L5", "J6:L8", "IOWs desviadas", "IOWs desviadas", card_int),
        ("M5:P5", "M6:P8", "Evidencia Bayes E(t)", "Evidencia Bayes", card_num),
    ]
    for label_rng, value_rng, label, field, fmt in cards:
        ws.merge_range(label_rng, label, card_label)
        ws.merge_range(value_rng, "", fmt)
        cell = value_rng.split(":")[0]
        ws.write_formula(cell, lookup_formula(field), fmt, field_cache.get(field, ""))

    # Segunda fila de KPI.
    cards2 = [
        ("A10:C10", "A11:C13", "Velocidad corrosión (mpy)", "Velocidad de corrosión (mpy)", card_num),
        ("D10:F10", "D11:F13", "Vida remanente (años)", "Vida remanente (años)", card_num),
        ("G10:I10", "G11:I13", "Fecha límite RVO", "Fecha límite RVO", card_text),
        ("J10:L10", "J11:L13", "Fecha programada", "Fecha programada", card_text),
        ("M10:P10", "M11:P13", "Presupuesto mantenimiento (S/)", "Presupuesto de mantenimiento (S/)", card_money),
    ]
    for label_rng, value_rng, label, field, fmt in cards2:
        ws.merge_range(label_rng, label, card_label)
        ws.merge_range(value_rng, "", fmt)
        cell = value_rng.split(":")[0]
        cached = field_cache.get(field, "")
        if isinstance(cached, pd.Timestamp): cached = cached.strftime("%d/%m/%Y")
        formula = lookup_formula(field)
        # Para fechas usamos TEXT para que el dashboard muestre una fecha legible.
        if field in {"Fecha límite RVO", "Fecha programada"}:
            formula = f'=IFERROR(TEXT({lookup_formula(field)[1:]},"dd/mm/yyyy"),"—")'
        ws.write_formula(cell, formula, fmt, cached if cached not in [None, ""] else "—")

    # Acción / prioridad / estado.
    ws.merge_range("A15:P15", "DECISIÓN PRESCRIPTIVA DEL TRAMO SELECCIONADO", section_fmt)
    decision_fields = [
        ("A16", "Acción recomendada"), ("A17", "Prioridad"), ("A18", "Estado de programación"),
        ("A19", "Costo económico total estimado (S/)"), ("A20", "Horas-hombre estimadas"),
        ("A21", "Utilización planificada de recursos (%)"),
    ]
    for r, (cell, field) in enumerate(decision_fields, start=16):
        ws.write(r-1, 0, field, header_fmt)
        ws.merge_range(r-1, 1, r-1, 5, "", cell_fmt)
        fmt = money_fmt if "Costo" in field else (percent_fmt if "%" in field else cell_fmt)
        ws.write_formula(r-1, 1, lookup_formula(field), fmt, field_cache.get(field, ""))

    # Matrices históricas ocultas + tabla auxiliar dinámica.
    matrix_start = hidden_start_col + len(per_pipe.columns) + 3
    metric_blocks = {}
    for metric in ["PoF_prior", "PoF_posterior", "Riesgo_DRBI", "E_t", "N_IOW_DESVIADAS"]:
        p = metric_pivots[metric]
        start_col = matrix_start
        metric_blocks[metric] = start_col
        ws.write(0, start_col, "Semana")
        for j, tid in enumerate(tramos, start=1): ws.write(0, start_col+j, tid)
        for i, week in enumerate(weeks, start=1):
            ws.write(i, start_col, week)
            for j, tid in enumerate(tramos, start=1):
                val = p.loc[week, tid]
                if pd.isna(val): ws.write_blank(i, start_col+j, None)
                else: ws.write_number(i, start_col+j, float(val))
        matrix_start += len(tramos) + 3

    helper_col = matrix_start
    helper_headers = ["Semana", "PoF inicial", "PoF actualizada", "Riesgo DRBI", "Evidencia E(t)", "IOWs desviadas"]
    for j, h in enumerate(helper_headers): ws.write(0, helper_col+j, h)
    for i, week in enumerate(weeks, start=1):
        ws.write(i, helper_col, week)
        for j, metric in enumerate(["PoF_prior", "PoF_posterior", "Riesgo_DRBI", "E_t", "N_IOW_DESVIADAS"], start=1):
            start_col = metric_blocks[metric]
            start_letter = xlsxwriter.utility.xl_col_to_name(start_col)
            first_metric_letter = xlsxwriter.utility.xl_col_to_name(start_col+1)
            last_metric_letter = xlsxwriter.utility.xl_col_to_name(start_col+len(tramos))
            formula = f'=IFERROR(INDEX(${first_metric_letter}$2:${last_metric_letter}${len(weeks)+1},ROW()-1,MATCH($B$3,${first_metric_letter}$1:${last_metric_letter}$1,0)),NA())'
            ws.write_formula(i, helper_col+j, formula)

    hcol = [xlsxwriter.utility.xl_col_to_name(helper_col+i) for i in range(6)]
    start_data = 2; end_data = len(weeks)+1
    # Gráfico PoF.
    c1 = wb.add_chart({"type": "line"})
    c1.add_series({"name": "PoF inicial", "categories": f"='Dashboard_KPI'!${hcol[0]}${start_data}:${hcol[0]}${end_data}", "values": f"='Dashboard_KPI'!${hcol[1]}${start_data}:${hcol[1]}${end_data}", "line": {"color": "#8A99AB", "dash_type": "dash"}})
    c1.add_series({"name": "PoF actualizada", "categories": f"='Dashboard_KPI'!${hcol[0]}${start_data}:${hcol[0]}${end_data}", "values": f"='Dashboard_KPI'!${hcol[2]}${start_data}:${hcol[2]}${end_data}", "line": {"color": BLUE, "width": 2.25}, "marker": {"type": "circle", "size": 4}})
    c1.set_title({"name": "Evolución de PoF del tramo seleccionado"}); c1.set_y_axis({"name": "Probabilidad", "min": 0}); c1.set_x_axis({"name": "Semana"}); c1.set_legend({"position": "bottom"}); c1.set_style(10)
    ws.insert_chart("H16", c1, {"x_scale": 1.22, "y_scale": 1.15})

    # Gráfico DRBI.
    c2 = wb.add_chart({"type": "line"})
    c2.add_series({"name": "Riesgo DRBI", "categories": f"='Dashboard_KPI'!${hcol[0]}${start_data}:${hcol[0]}${end_data}", "values": f"='Dashboard_KPI'!${hcol[3]}${start_data}:${hcol[3]}${end_data}", "line": {"color": RED, "width": 2.25}, "marker": {"type": "circle", "size": 4}})
    c2.set_title({"name": "Evolución del riesgo DRBI"}); c2.set_y_axis({"name": "PoF × CoF", "min": 0}); c2.set_x_axis({"name": "Semana"}); c2.set_legend({"none": True}); c2.set_style(10)
    ws.insert_chart("A24", c2, {"x_scale": 1.22, "y_scale": 1.12})

    # Gráfico evidencia/IOW combinado.
    c3 = wb.add_chart({"type": "column"})
    c3.add_series({"name": "IOWs desviadas", "categories": f"='Dashboard_KPI'!${hcol[0]}${start_data}:${hcol[0]}${end_data}", "values": f"='Dashboard_KPI'!${hcol[5]}${start_data}:${hcol[5]}${end_data}", "fill": {"color": AMBER}, "border": {"none": True}})
    line = wb.add_chart({"type": "line"})
    line.add_series({"name": "Evidencia E(t)", "categories": f"='Dashboard_KPI'!${hcol[0]}${start_data}:${hcol[0]}${end_data}", "values": f"='Dashboard_KPI'!${hcol[4]}${start_data}:${hcol[4]}${end_data}", "y2_axis": True, "line": {"color": NAVY, "width": 2}})
    c3.combine(line); c3.set_title({"name": "IOWs desviadas y evidencia Bayes"}); c3.set_x_axis({"name": "Semana"}); c3.set_y_axis({"name": "N.º IOWs"}); c3.set_y2_axis({"name": "E(t)", "min": 0, "max": 1}); c3.set_legend({"position": "bottom"}); c3.set_style(10)
    ws.insert_chart("H34", c3, {"x_scale": 1.22, "y_scale": 1.12})

    # Tabla KPI completa por tramo seleccionado.
    ws.merge_range("A44:P44", "KPI DETALLADOS DEL TRAMO SELECCIONADO", section_fmt)
    detail_fields = [
        "PoF inicial", "PoF actualizada", "Variación de PoF", "CoF", "Riesgo DRBI", "Nivel de riesgo",
        "IOWs desviadas", "Magnitud M(t)", "Persistencia D(t)", "Frecuencia F(t)", "Evidencia Bayes",
        "Espesor actual (mm)", "Velocidad de corrosión (mpy)", "Vida remanente (años)", "Acción recomendada",
        "Prioridad", "Fecha límite RVO", "Estado de programación", "Fecha programada",
        "Presupuesto de mantenimiento (S/)", "Impacto operativo estimado (S/)", "Costo económico total estimado (S/)",
        "Horas-hombre estimadas", "Horas de intervención estimadas", "Utilización planificada de recursos (%)",
    ]
    ws.write_row("A45", ["KPI", "Resultado", "Interpretación"], header_fmt)
    for i, field in enumerate(detail_fields, start=46):
        ws.write(i-1, 0, field, cell_fmt)
        fmt = money_fmt if "(S/)" in field else (date_fmt if "Fecha" in field else (percent_fmt if "%" in field else (num_fmt if field not in {"Nivel de riesgo", "Acción recomendada", "Prioridad", "Estado de programación"} else cell_fmt)))
        formula = lookup_formula(field)
        cached = field_cache.get(field, "")
        if isinstance(cached, pd.Timestamp):
            ws.write_formula(i-1, 1, formula, date_fmt, cached.to_pydatetime())
        else:
            ws.write_formula(i-1, 1, formula, fmt, cached if not pd.isna(cached) else "")
        ws.write(i-1, 2, KPI_TRAMO_EXPLICACION.get(field, "Indicador técnico específico del tramo seleccionado."), cell_fmt)

    # Resumen de todos los tramos (sin mezclar sus KPI).
    summary_row = 46 + len(detail_fields) + 2
    ws.merge_range(summary_row-1, 0, summary_row-1, 15, "RESUMEN COMPARATIVO DE LOS TRAMOS DEL TIPO SELECCIONADO", section_fmt)
    summary_cols = ["Tramo", "Servicio / fluido", "PoF actualizada", "Riesgo DRBI", "Nivel de riesgo", "Acción recomendada", "Prioridad", "Fecha límite RVO", "Estado de programación", "Fecha programada"]
    ws.write_row(summary_row, 0, summary_cols, header_fmt)
    for i, (_, row) in enumerate(per_pipe[summary_cols].iterrows(), start=summary_row+1):
        for j, col in enumerate(summary_cols):
            val = row[col]
            if isinstance(val, pd.Timestamp): ws.write_datetime(i, j, val.to_pydatetime(), date_fmt)
            elif pd.isna(val): ws.write_blank(i, j, None, cell_fmt)
            elif col == "Nivel de riesgo": ws.write(i, j, val, risk_fmt.get(str(val), cell_fmt))
            elif col in {"PoF actualizada", "Riesgo DRBI"}: ws.write_number(i, j, float(val), num_fmt)
            else: ws.write(i, j, val, cell_fmt)

    ws.set_column("A:A", 30); ws.set_column("B:B", 20); ws.set_column("C:C", 60)
    ws.set_column("D:P", 15)
    # Ocultar todo el backend del dashboard.
    ws.set_column(hidden_start_col, helper_col+5, None, None, {"hidden": True})
    ws.freeze_panes(3, 0)

    # ------------------------------------------------------------------
    # HOJA 2: Programa_Mantenimiento
    # ------------------------------------------------------------------
    wp = wb.add_worksheet("Programa_Mantenimiento"); wp.hide_gridlines(2)
    wp.merge_range("A1:X1", f"PROGRAMA DE MANTENIMIENTO OPTIMIZADO – TUBERÍAS DE {tipo.upper()}", title_fmt)
    wp.merge_range("A2:X2", f"Fechas trazables desde la última evidencia ({pd.Timestamp(result['fecha_corte_analisis']):%d/%m/%Y}); MILP respeta lead time, RVO, ventanas, recursos y presupuesto.", subtitle_fmt)
    basis = temporal_basis_table(result)
    wp.merge_range("A4:X4", "BASE Y TRAZABILIDAD DE LAS FECHAS", section_fmt)
    # Escribir DataFrame simple.
    def write_df(sheet, df, row0, col0):
        if df is None or df.empty: return row0
        sheet.write_row(row0, col0, list(df.columns), header_fmt)
        for i, (_, r) in enumerate(df.iterrows(), start=row0+1):
            for j, col in enumerate(df.columns):
                v = r[col]
                if isinstance(v, pd.Timestamp) and not pd.isna(v): sheet.write_datetime(i, col0+j, v.to_pydatetime(), date_fmt)
                elif pd.isna(v): sheet.write_blank(i, col0+j, None, cell_fmt)
                elif "(S/)" in str(col): sheet.write_number(i, col0+j, float(v), money_fmt)
                elif isinstance(v, (int, float, np.integer, np.floating)): sheet.write_number(i, col0+j, float(v), num_fmt)
                else: sheet.write(i, col0+j, v, cell_fmt)
        return row0 + len(df)
    end_basis = write_df(wp, basis, 4, 0)
    pr = end_basis + 3
    wp.merge_range(pr, 0, pr, 23, "PROGRAMACIÓN MILP – FECHA, RESTRICCIONES Y FUNDAMENTO", section_fmt)
    end_prog = write_df(wp, program, pr+1, 0)
    rr = end_prog + 3
    wp.merge_range(rr, 0, rr, 23, "SEGUIMIENTO Y REPROGRAMACIÓN (ROLLING HORIZON)", section_fmt)
    write_df(wp, rolling, rr+1, 0)
    wp.freeze_panes(pr+2, 0)
    wp.set_column(0, 23, 18)
    for c in [3, 9, 17]: wp.set_column(c, c, 42)

    # ------------------------------------------------------------------
    # HOJA 3: Trazabilidad_7_Capas
    # ------------------------------------------------------------------
    wt = wb.add_worksheet("Trazabilidad_7_Capas"); wt.hide_gridlines(2)
    wt.merge_range("A1:Q1", "TRAZABILIDAD DEL MODELO INTEGRADO – 7 CAPAS", title_fmt)
    wt.merge_range("A2:Q2", "La Capa 7 cierra el modelo con KPI individualizados por tramo y retroalimentación.", subtitle_fmt)
    layer_map = pd.DataFrame([
        [1, "IIoT / ETL", "Datos operativos, laboratorio, inspección y mantenimiento", "Valida, limpia, ordena e integra la información.", "Base confiable por tramo y fecha"],
        [2, "IOWs", "Base integrada + límites operativos", "Detecta desviaciones y construye evidencia.", "Magnitud, persistencia, frecuencia y E(t)"],
        [3, "Bayes", "PoF inicial + evidencia IOW", "Actualiza la probabilidad de falla.", "PoF actualizada por tramo"],
        [4, "DRBI", "PoF actualizada + CoF", "Calcula el riesgo dinámico.", "Riesgo y nivel por tramo"],
        [5, "MDP-RVO", "Riesgo y condición", "Selecciona acción, prioridad y plazo.", "Acción + fecha límite por tramo"],
        [6, "MILP", "Acción + ventana + recursos + presupuesto", "Asigna fecha factible.", "Programa por tramo"],
        [7, "Evaluación y KPI", "Resultados de las capas anteriores", "Calcula KPI individuales y habilita comparación/reprogramación.", "Dashboard final por tubería/tramo"],
    ], columns=["Capa","Etapa","Entrada","Qué realiza","Salida"])
    wt.merge_range("A4:E4", "MAPA DE LAS 7 CAPAS", section_fmt)
    write_df(wt, layer_map, 4, 0)
    wt.merge_range("A15:Q15", "KPI ACTUALES POR TRAMO – SIN MEZCLAR TUBERÍAS", section_fmt)
    trace_cols = ["Tramo", "Servicio / fluido", "PoF actualizada", "Variación de PoF", "Riesgo DRBI", "Nivel de riesgo", "IOWs desviadas", "Evidencia Bayes", "Acción recomendada", "Prioridad", "Fecha límite RVO", "Estado de programación", "Fecha programada", "Presupuesto de mantenimiento (S/)", "Costo económico total estimado (S/)", "Horas-hombre estimadas", "Utilización planificada de recursos (%)"]
    write_df(wt, per_pipe[trace_cols], 15, 0)
    wt.freeze_panes(16,0); wt.set_column(0,16,18); wt.set_column(1,1,22); wt.set_column(8,8,42)

    wb.close()
    return out_path

# =============================================================================
# 23. PLANTILLA XML
# =============================================================================

def xml_template(path: str, tipo: str = "Proceso") -> str:
    data = demo_dataset(semanas=2)
    data = data[data["TIPO_TUBERIA"].eq(tipo)].head(6)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<datos>"]
    for _, row in data.iterrows():
        lines.append("  <registro>")
        for col, value in row.items():
            if isinstance(value, pd.Timestamp):
                value = value.isoformat()
            safe = html.escape(str(value))
            lines.append(f"    <{col}>{safe}</{col}>")
        lines.append("  </registro>")
    lines.append("</datos>")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(f"DSS {VERSION_DSS} - validación interna")
    print(self_test().to_string(index=False))
