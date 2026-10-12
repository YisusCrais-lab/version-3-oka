# DSS de Integridad y Mantenimiento de Tuberías

Modelo integrado:

**IIoT/ETL → IOWs → Bayes → DRBI → MDP-RVO → MILP → KPI / Rolling Horizon**

## Tipos de tubería
- Submarina
- Transferencia
- Proceso

## Archivo principal de Streamlit
`app.py`

## Ejecución local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicación en Streamlit Community Cloud
1. Subir esta carpeta a un repositorio de GitHub.
2. Crear una aplicación nueva en Streamlit Community Cloud.
3. Seleccionar el repositorio.
4. En **Main file path**, indicar `app.py`.
5. Desplegar.

## Funciones principales
- Selección obligatoria del tipo de tubería antes del análisis.
- Carga de XML, XLSX, XLS y CSV.
- Ingreso manual de datos.
- Carga de ventanas reales o generación provisional.
- Siete capas del DSS.
- KPI independientes por número/ID de tubería.
- Dashboard y gráficos personalizados por tubería.
- Política de mantenimiento y política de programación dentro de la plataforma.
- Programa MILP con trazabilidad de fechas.
- Reporte Excel final sin políticas.
- Base SQLite local para registrar ejecuciones y KPI por tramo.

## Fechas de programación
1. Fecha de corte = última evidencia disponible.
2. Primera fecha posible = fecha de corte + lead time técnico.
3. Fecha límite RVO = fecha de corte + plazo RVO.
4. Compatibilidad con ventana operativa.
5. Restricciones de recursos y presupuesto.
6. Asignación final mediante MILP.

## Nota de uso
Los parámetros de demostración deben ser reemplazados por límites, costos,
ventanas, capacidades y reglas aprobadas por la organización antes de un uso operativo.
