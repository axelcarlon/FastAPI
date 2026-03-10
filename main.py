from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from typing import List
import json
import os
import logging
import asyncio # <--- AÑADIDO PARA LA PAUSA DE SEGURIDAD

# ---------------- CONFIGURACIÓN FORENSE Y LOGS ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI(title="AuditorIA Core - Big 4 Standard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    logging.critical("GEMINI_API_KEY no detectada. El motor no podrá arrancar.")

client = genai.Client(api_key=api_key)

# ---------------- MANEJADOR GLOBAL DE EXCEPCIONES ----------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Fallo en {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "codigo": "SYS_ERR", "mensaje": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"} # <--- CORRECCIÓN DEL BUG CORS
    )

# ---------------- MODELOS DE DATOS ----------------
class SolicitudAnalisis(BaseModel):
    tipo_documento: str
    error_detectado: str
    descripcion_concepto: str

class MensajeChat(BaseModel):
    mensaje: str
    historial: list

class SolicitudRiskScore(BaseModel):
    puntaje: int
    factores: str

class DatosConciliacion(BaseModel):
    bancos: list
    facturas: list

class SolicitudTasaEfectiva(BaseModel):
    ingresos: float
    isr_pagado: float
    sector: str

class SolicitudAduana(BaseModel):
    conceptos: list

# ---------------- MOTORES DE ESTABILIDAD (BLINDAJE) ----------------
def extraer_json(texto: str):
    """Búsqueda matemática de envolventes JSON. Inmune a texto basura de la IA."""
    try:
        t = texto.strip()
        idx_obj = t.find('{')
        idx_arr = t.find('[')
        
        if idx_obj == -1 and idx_arr == -1:
            raise ValueError("Delimitadores JSON ausentes en la respuesta.")
        
        if idx_obj != -1 and (idx_arr == -1 or idx_obj < idx_arr):
            start = idx_obj
            end = t.rfind('}') + 1
        else:
            start = idx_arr
            end = t.rfind(']') + 1
            
        return json.loads(t[start:end])
    except Exception as e:
        logging.error(f"Corrupción de datos IA: {texto}")
        raise ValueError("Estructura matemática devuelta por la IA es ilegible.")

def reparar_mime(filename: str, current_mime: str) -> str:
    """Interceptor de cabeceras para forzar la compatibilidad binaria de Windows/Edge."""
    if not current_mime or current_mime == "application/octet-stream":
        ext = filename.split('.')[-1].lower() if filename else ""
        if ext == "pdf": return "application/pdf"
        elif ext in ["jpg", "jpeg"]: return "image/jpeg"
        elif ext in ["png"]: return "image/png"
    return current_mime if current_mime else "application/pdf"

# ---------------- ENDPOINTS DE OPERACIÓN E INTELIGENCIA ----------------

@app.get("/ping")
async def ping():
    return {"status": "ok", "motor": "AuditorIA Core Activo"}

@app.post("/api/asesor-ia")
async def analizar_discrepancia(solicitud: SolicitudAnalisis):
    prompt = f"""
    Actúa como Auditor Fiscal Forense.
    Paso 1: Analiza el documento '{solicitud.tipo_documento}' y el concepto '{solicitud.descripcion_concepto}'.
    Paso 2: Evalúa el error detectado: '{solicitud.error_detectado}'.
    Paso 3: Cruza la inconsistencia con la LISR y el CFF vigentes en México.
    Paso 4: Redacta un dictamen técnico de 3 párrafos. Sé directo, crudo y usa lenguaje jurídico-contable.
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(tools=[{"google_search": {}}], temperature=0.1)
    )
    return {"analisis_legal": response.text.strip()}

@app.post("/api/chat")
async def chat_asesor(datos: MensajeChat):
    instruccion = "Eres Paula, Asesor Fiscal IA. Respuestas frías, quirúrgicas, directas al punto. Prohibido saludar o pedir disculpas. Basado estrictamente en LISR, LIVA y CFF."
    contents = [{"role": "user", "parts": [{"text": instruccion}]}, {"role": "model", "parts": [{"text": "Entendido. Procesando con rigor legal."}]}]
    for msj in datos.historial:
        role = "user" if msj["rol"] == "usuario" else "model"
        contents.append({"role": role, "parts": [{"text": msj["texto"]}]})
    contents.append({"role": "user", "parts": [{"text": datos.mensaje}]})
    
    response = client.models.generate_content(model='gemini-2.5-flash', contents=contents, config=types.GenerateContentConfig(tools=[{"google_search": {}}]))
    return {"respuesta": response.text.strip()}

@app.post("/api/ocr-fiscal")
async def ocr_fiscal(archivos: List[UploadFile] = File(...)):
    resultados = []
    for archivo in archivos:
        file_bytes = await archivo.read()
        mime = reparar_mime(archivo.filename, archivo.content_type)
        prompt_ocr = """
        Ejecuta extracción OCR Forense.
        Paso 1: Evalúa la calidad. Si es totalmente ilegible, devuelve {"error": "ERROR_CALIDAD_IMAGEN"}.
        Paso 2: Extrae los montos. Si detectas palabras como "Propina" o "Bebidas Alcohólicas", clasifícalos pero asegúrate de extraer el subtotal y total exactos.
        Paso 3: Devuelve ÚNICAMENTE un JSON con esta estructura:
        {"rfc_emisor": "str", "rfc_receptor": "str", "subtotal": float, "total_traslados": float, "total_retenciones": float, "total": float, "conceptos": [{"descripcion": "str", "importe": float}]}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt_ocr],
            config=types.GenerateContentConfig(temperature=0.0)
        )
        try:
            datos = extraer_json(response.text)
            if "error" in datos:
                resultados.append({"archivo": archivo.filename, "status": "error_calidad", "mensaje": "Imagen rechazada por el motor de visión."})
            else:
                resultados.append({"archivo": archivo.filename, "status": "success", "datos": datos})
        except:
            resultados.append({"archivo": archivo.filename, "status": "error", "mensaje": "Vectorización fallida."})
            
        await asyncio.sleep(2.5) # <--- PAUSA DE SEGURIDAD PARA EVITAR BANEO DE GOOGLE
            
    return resultados[0]["datos"] if len(resultados) == 1 and resultados[0]["status"] == "success" else {"status": "success_lote", "resultados_lote": resultados}

@app.post("/api/risk-score-ia")
async def risk_score_ia(solicitud: SolicitudRiskScore):
    prompt = f"""
    Aplica Predictive Auditing basado en el modelo Ra = Σ(wi * ki).
    Puntaje: {solicitud.puntaje}/100. Factores: {solicitud.factores}.
    Paso 1: Evalúa la gravedad de los factores.
    Paso 2: Determina el porcentaje exacto de probabilidad de facultades de comprobación (Art. 42 CFF) a 90 días.
    Paso 3: Redacta un dictamen corporativo preventivo de 2 párrafos.
    """
    response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt, config=types.GenerateContentConfig(temperature=0.1))
    return {"analisis_legal": response.text.strip()}

@app.post("/api/conciliacion-fuzzy")
async def conciliacion_fuzzy(datos: DatosConciliacion):
    prompt = """
    Motor Fuzzy Matching Semántico.
    Paso 1: Analiza la matriz de bancos y la matriz de facturas.
    Paso 2: Cruza importes exactos. Si hay variaciones de centavos, usa análisis de fechas y conceptos.
    Paso 3: Asigna un nivel_confianza de 0 a 100.
    Paso 4: Devuelve SOLO JSON con matches > 60:
    {"conciliados": [{"id_banco": "str", "uuid_factura": "str", "razonamiento": "str", "nivel_confianza": int}]}
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=[prompt, json.dumps({"bancos": datos.bancos, "facturas": datos.facturas})],
        config=types.GenerateContentConfig(temperature=0.0)
    )
    return extraer_json(response.text)

@app.post("/api/materialidad")
async def validar_materialidad(contrato: UploadFile = File(...), datos_xml: str = Form(...)):
    file_bytes = await contrato.read()
    mime = reparar_mime(contrato.filename, contrato.content_type)
    prompt = f"""
    Auditoría Forense Art. 5-A.
    Paso 1: Lee el Contrato adjunto.
    Paso 2: Lee los metadatos de la Factura: {datos_xml}.
    Paso 3: Busca discrepancias en montos, fechas, o carencia de entregables específicos (cláusulas huecas).
    Paso 4: Devuelve SOLO JSON:
    {{"nivel_riesgo": "BAJO"|"MEDIO"|"ALTO", "porcentaje_coincidencia": int, "dictamen_defensa": "str", "hallazgos": ["str"]}}
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt],
        config=types.GenerateContentConfig(temperature=0.1)
    )
    return extraer_json(response.text)

@app.post("/api/tasa-efectiva")
async def evaluar_tasa_efectiva(datos: SolicitudTasaEfectiva):
    tasa = (datos.isr_pagado / datos.ingresos) * 100 if datos.ingresos > 0 else 0
    prompt = f"""
    Inteligencia Sectorial SAT.
    Paso 1: Evalúa la tasa de {tasa:.2f}% para el sector '{datos.sector}'.
    Paso 2: Compara con el Plan Maestro de Fiscalización.
    Paso 3: Devuelve SOLO JSON: 
    {{"tasa_calculada": {tasa:.2f}, "nivel_riesgo": "BAJO"|"MEDIO"|"ALTO", "dictamen_sectorial": "str"}}
    """
    response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt, config=types.GenerateContentConfig(temperature=0.1))
    return extraer_json(response.text)

@app.post("/api/auditoria-activos")
async def auditoria_activos(foto: UploadFile = File(...), datos_xml: str = Form(...)):
    file_bytes = await foto.read()
    mime = reparar_mime(foto.filename, foto.content_type)
    prompt = f"""
    Peritaje de Materialidad Visual.
    Paso 1: Identifica el objeto en la imagen y su valor estimado de mercado.
    Paso 2: Cruza con los datos del XML: {datos_xml}.
    Paso 3: Si el XML dice "Servicios" y la foto es un bien físico, marca discrepancia crítica.
    Paso 4: Devuelve SOLO JSON:
    {{"coincidencia": bool, "datos_extraidos_foto": "str", "dictamen_materialidad": "str"}}
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt],
        config=types.GenerateContentConfig(temperature=0.1)
    )
    return extraer_json(response.text)

@app.post("/api/precios-aduana")
async def precios_aduana(datos: SolicitudAduana):
    prompt = f"""
    Auditoría ANAM.
    Paso 1: Revisa conceptos XML: {json.dumps(datos.conceptos)}.
    Paso 2: Si no tienen fracción arancelaria o pedimento, el riesgo es BAJO. Si lo tienen, busca subvaluación.
    Paso 3: Devuelve SOLO JSON:
    {{"nivel_riesgo_aduanero": "BAJO"|"MEDIO"|"ALTO", "analisis_valoracion": "str"}}
    """
    response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt, config=types.GenerateContentConfig(temperature=0.1))
    return extraer_json(response.text)

@app.post("/api/prueba-servicio")
async def prueba_servicio(evidencia: UploadFile = File(...), datos_xml: str = Form(...)):
    file_bytes = await evidencia.read()
    mime = reparar_mime(evidencia.filename, evidencia.content_type)
    prompt = f"""
    Cruce probatorio Art. 69-B.
    Paso 1: Analiza la evidencia documental.
    Paso 2: Crúzala con la descripción de la factura: {datos_xml}.
    Paso 3: Devuelve SOLO JSON:
    {{"sustancia_economica_comprobada": bool, "nivel_riesgo_efos": "BAJO"|"MEDIO"|"ALTO", "dictamen_evidencia": "str"}}
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt],
        config=types.GenerateContentConfig(temperature=0.1)
    )
    return extraer_json(response.text)

@app.post("/api/viaticos-geo")
async def viaticos_geo(datos_xml: str = Form(...)):
    prompt = f"""
    Auditoría Art. 28 LISR (Viáticos).
    Paso 1: Extrae CP Emisor y CP Receptor del XML: {datos_xml}.
    Paso 2: Calcula distancia lineal. Si es menor a 50km, deducible = false.
    Paso 3: Revisa monto contra topes (750 MXN / 1500 MXN).
    Paso 4: Devuelve SOLO JSON:
    {{"deducible": bool, "distancia_estimada_km": float, "limite_respetado": bool, "dictamen_viaticos": "str"}}
    """
    response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt, config=types.GenerateContentConfig(temperature=0.1))
    return extraer_json(response.text)

@app.post("/api/defensa-legal")
async def defensa_legal(documento: UploadFile = File(...)):
    file_bytes = await documento.read()
    mime = reparar_mime(documento.filename, documento.content_type)
    prompt = """
    Eres un Magistrado Fiscal.
    Paso 1: Lee el requerimiento o multa del SAT adjunto.
    Paso 2: Identifica el fundamento legal atacado.
    Paso 3: Redacta un Recurso de Revocación u Oficio de Aclaración impecable en formato texto estructurado (listo para copiar a Word).
    Paso 4: Excluye marcas de agua o texto residual de la IA.
    """
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt],
        config=types.GenerateContentConfig(tools=[{"google_search": {}}], temperature=0.1)
    )
    return {"oficio_legal": response.text.strip()}

@app.post("/api/banco-csv")
async def banco_csv(documento: UploadFile = File(...)):
    file_bytes = await documento.read()
    mime = reparar_mime(documento.filename, documento.content_type)
    prompt = """
    Paso 1: Extrae todas las tablas financieras del estado de cuenta.
    Paso 2: Devuelve la información ÚNICAMENTE en texto CSV con estas columnas: Fecha (DD/MM/AAAA),Concepto,Cargo,Abono,Saldo.
    Paso 3: Ningún texto fuera del CSV.
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt],
        config=types.GenerateContentConfig(temperature=0.0)
    )
    return {"csv_data": response.text.strip()}

@app.post("/api/analista-csf")
async def analista_csf(documento: UploadFile = File(...)):
    file_bytes = await documento.read()
    mime = reparar_mime(documento.filename, documento.content_type)
    prompt = """
    Extracción KYC/KYV Forense.
    Paso 1: Determina si es Constancia de Situación Fiscal o 32-D.
    Paso 2: Extrae los datos solicitados.
    Paso 3: Devuelve SOLO JSON:
    {
        "rfc": "str", "razon_social": "str", "regimen_fiscal": ["str"], 
        "codigo_postal": "str", "tipo_documento": "CSF/32-D", 
        "estatus_cumplimiento": "ACTIVO/POSITIVA/NEGATIVA", "alertas": ["str"]
    }
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt],
        config=types.GenerateContentConfig(temperature=0.0)
    )
    return extraer_json(response.text)

# ---------------- PROTOCOLO FUTURO: SUITE PDF ----------------
@app.post("/api/pdf-studio")
async def pdf_studio_router(documento: UploadFile = File(...), accion: str = Form(...)):
    """Endpoint base preparado para la Fase 3: Escisión, Censura y Fusión de PDFs."""
    return {"status": "standby", "mensaje": "Módulo PDF Studio preparado para inyección lógica."}
