from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from typing import List
import json
import os
import re
import logging

# Registro de errores para depuración en Render
logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

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

# ---------------- MOTORES DE ESTABILIDAD (SÚPER-FUNCIONES) ----------------

def extraer_json(texto: str):
    """Limpia y extrae JSON de forma quirúrgica. Evita Errores 500 por formato."""
    try:
        # Elimina bloques markdown si existen
        texto_limpio = re.sub(r'```json\s*|```', '', texto).strip()
        # Busca la estructura de llaves para ignorar texto residual de la IA
        inicio = texto_limpio.find('{')
        fin = texto_limpio.rfind('}')
        if inicio != -1 and fin != -1:
            return json.loads(texto_limpio[inicio:fin+1])
        return json.loads(texto_limpio)
    except Exception as e:
        logging.error(f"Fallo al decodificar IA: {texto}")
        raise ValueError("La respuesta de la IA no es un JSON procesable.")

def reparar_mime(filename: str, current_mime: str) -> str:
    """Previene que Google rechace archivos PDF/Imágenes desde Windows/Edge."""
    if not current_mime or current_mime == "application/octet-stream":
        ext = filename.split('.')[-1].lower() if filename else ""
        if ext == "pdf": return "application/pdf"
        elif ext in ["jpg", "jpeg"]: return "image/jpeg"
        elif ext in ["png"]: return "image/png"
    return current_mime if current_mime else "application/pdf"

# ---------------- ENDPOINTS ----------------

@app.get("/ping")
async def ping():
    return {"status": "ok", "mensaje": "Servidor AuditorIA 2.0 Activo"}

@app.post("/api/asesor-ia")
async def analizar_discrepancia(solicitud: SolicitudAnalisis):
    try:
        prompt = f"""
        Eres el motor IA de AuditorIA, experto en LISR, LIVA y CFF. 
        Analiza: Documento: {solicitud.tipo_documento} | Concepto: {solicitud.descripcion_concepto} | Error: {solicitud.error_detectado}
        Cita el artículo de ley aplicable. No uses relleno.
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(tools=[{"google_search": {}}], temperature=0.2)
        )
        return {"analisis_legal": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_asesor(datos: MensajeChat):
    try:
        instruccion = "Eres Paula, Asesor Fiscal de AuditorIA. Respuestas frías, directas, sin saludos, basadas en leyes vigentes."
        contents = [{"role": "user", "parts": [{"text": instruccion}]}, {"role": "model", "parts": [{"text": "Entendido."}]}]
        for msj in datos.historial:
            role = "user" if msj["rol"] == "usuario" else "model"
            contents.append({"role": role, "parts": [{"text": msj["texto"]}]})
        contents.append({"role": "user", "parts": [{"text": datos.mensaje}]})
        
        response = client.models.generate_content(model='gemini-2.5-flash', contents=contents, config=types.GenerateContentConfig(tools=[{"google_search": {}}]))
        return {"respuesta": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ocr-fiscal")
async def ocr_fiscal(archivos: List[UploadFile] = File(...)):
    try:
        resultados = []
        for archivo in archivos:
            file_bytes = await archivo.read()
            mime = reparar_mime(archivo.filename, archivo.content_type)
            prompt_ocr = """Extrae JSON estricto: {"rfc_emisor": "str", "rfc_receptor": "str", "subtotal": float, "total_traslados": float, "total_retenciones": float, "total": float, "conceptos": [{"descripcion": "str", "importe": float}]}"""
            response = client.models.generate_content(model='gemini-2.5-flash', contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt_ocr])
            try:
                datos = extraer_json(response.text)
                resultados.append({"archivo": archivo.filename, "status": "success", "datos": datos})
            except:
                resultados.append({"archivo": archivo.filename, "status": "error", "mensaje": "Falla al estructurar datos."})
        return resultados[0]["datos"] if len(resultados) == 1 else {"status": "success_lote", "resultados_lote": resultados}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/risk-score-ia")
async def risk_score_ia(solicitud: SolicitudRiskScore):
    try:
        prompt = f" Predictive Auditing. Puntaje: {solicitud.puntaje}/100. Factores: {solicitud.factores}. Dictamina probabilidad de auditoría Art. 42 CFF."
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt, config=types.GenerateContentConfig(temperature=0.2))
        return {"analisis_legal": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conciliacion-fuzzy")
async def conciliacion_fuzzy(datos: DatosConciliacion):
    try:
        prompt = "Fuzzy Matching. Relaciona bancos vs facturas. Devuelve SOLO JSON: {'conciliados': [{'id_banco': 'str', 'uuid_factura': 'str', 'razonamiento': 'str', 'nivel_confianza': int}]}"
        response = client.models.generate_content(model='gemini-2.5-flash', contents=[prompt, json.dumps({"bancos": datos.bancos, "facturas": datos.facturas})])
        return extraer_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/materialidad")
async def validar_materialidad(contrato: UploadFile = File(...), datos_xml: str = Form(...)):
    try:
        file_bytes = await contrato.read()
        mime = reparar_mime(contrato.filename, contrato.content_type)
        prompt = f"Auditor Forense. Cruza Contrato vs XML: {datos_xml}. Devuelve SOLO JSON."
        response = client.models.generate_content(model='gemini-2.5-flash', contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt])
        return extraer_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tasa-efectiva")
async def evaluar_tasa_efectiva(datos: SolicitudTasaEfectiva):
    try:
        tasa = (datos.isr_pagado / datos.ingresos) * 100 if datos.ingresos > 0 else 0
        prompt = f"Evalúa Tasa ISR {tasa:.2f}% para sector '{datos.sector}'. Compara con SAT. Devuelve SOLO JSON: {{'tasa_calculada': {tasa:.2f}, 'nivel_riesgo': 'BAJO/MEDIO/ALTO', 'dictamen_sectorial': 'str'}}"
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return extraer_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auditoria-activos")
async def auditoria_activos(foto: UploadFile = File(...), datos_xml: str = Form(...)):
    try:
        file_bytes = await foto.read()
        mime = reparar_mime(foto.filename, foto.content_type)
        prompt = f"Perito materialidad. Foto vs XML: {datos_xml}. Devuelve SOLO JSON."
        response = client.models.generate_content(model='gemini-2.5-flash', contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt])
        return extraer_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/precios-aduana")
async def precios_aduana(datos: SolicitudAduana):
    try:
        prompt = f"Agente Aduanal. Analiza conceptos: {json.dumps(datos.conceptos)}. Devuelve SOLO JSON."
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return extraer_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prueba-servicio")
async def prueba_servicio(evidencia: UploadFile = File(...), datos_xml: str = Form(...)):
    try:
        file_bytes = await evidencia.read()
        mime = reparar_mime(evidencia.filename, evidencia.content_type)
        prompt = f"Auditor SAT. Evidencia vs XML: {datos_xml}. Devuelve SOLO JSON."
        response = client.models.generate_content(model='gemini-2.5-flash', contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt])
        return extraer_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/viaticos-geo")
async def viaticos_geo(datos_xml: str = Form(...)):
    try:
        prompt = f"Auditor SAT. Analiza viáticos en XML: {datos_xml}. Devuelve SOLO JSON."
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        return extraer_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/defensa-legal")
async def defensa_legal(documento: UploadFile = File(...)):
    try:
        file_bytes = await documento.read()
        mime = reparar_mime(documento.filename, documento.content_type)
        prompt = "Abogado fiscal. Redacta defensa legal. Sin marcas de agua. Texto legal puro."
        response = client.models.generate_content(model='gemini-2.5-pro', contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt])
        return {"oficio_legal": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/banco-csv")
async def banco_csv(documento: UploadFile = File(...)):
    try:
        file_bytes = await documento.read()
        mime = reparar_mime(documento.filename, documento.content_type)
        prompt = "Extractor financiero. CSV: Fecha,Concepto,Cargo,Abono,Saldo"
        response = client.models.generate_content(model='gemini-2.5-flash', contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt])
        return {"csv_data": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analista-csf")
async def analista_csf(documento: UploadFile = File(...)):
    try:
        file_bytes = await documento.read()
        mime = reparar_mime(documento.filename, documento.content_type)
        prompt = "Analiza documento fiscal. Devuelve SOLO JSON: {'rfc':'','razon_social':'','regimen_fiscal':[],'codigo_postal':'','estatus_cumplimiento':'','alertas':[]}"
        response = client.models.generate_content(model='gemini-2.5-flash', contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt])
        return extraer_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
