from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from typing import List
import json
import os
import re

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

# ---------------- MOTORES DE ESTABILIDAD ----------------

def extraer_json(texto: str):
    """Limpia la respuesta de la IA y extrae el JSON puro de forma quirúrgica."""
    try:
        # Intento 1: Quitar etiquetas markdown
        texto_limpio = re.sub(r'```json\s*|```', '', texto).strip()
        # Intento 2: Buscar llaves si hay texto residual
        inicio = texto_limpio.find('{')
        fin = texto_limpio.rfind('}')
        if inicio != -1 and fin != -1:
            return json.loads(texto_limpio[inicio:fin+1])
        return json.loads(texto_limpio)
    except Exception as e:
        print(f"Error decodificando IA: {texto}")
        raise ValueError("La respuesta de la IA no contiene un JSON válido.")

def reparar_mime(filename: str, current_mime: str) -> str:
    """Previene que Google rechace archivos por detección errónea del navegador."""
    if not current_mime or current_mime == "application/octet-stream":
        ext = filename.split('.')[-1].lower() if filename else ""
        if ext == "pdf": return "application/pdf"
        elif ext in ["jpg", "jpeg"]: return "image/jpeg"
        elif ext in ["png"]: return "image/png"
    return current_mime if current_mime else "application/pdf"

# ---------------- ENDPOINTS GENERALES E IA ----------------

@app.get("/ping")
async def ping():
    return {"status": "ok", "mensaje": "Servidor AuditorIA 2.0 Activo"}

@app.post("/api/asesor-ia")
async def analizar_discrepancia(solicitud: SolicitudAnalisis):
    try:
        prompt = f"""
        Eres el motor IA de AuditorIA, experto en LISR, LIVA y CFF. 
        Analiza: Documento: {solicitud.tipo_documento} | Concepto: {solicitud.descripcion_concepto} | Error: {solicitud.error_detectado}
        Cita el artículo de ley aplicable para defensa o sanción. No uses relleno.
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.2
            )
        )
        return {"analisis_legal": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_asesor(datos: MensajeChat):
    try:
        instruccion = "Eres Paula, Asesor Fiscal Virtual de AuditorIA. Busca en internet las leyes fiscales vigentes (DOF, SAT) para responder. Respuestas frías, directas, sin saludos."
        contents = [{"role": "user", "parts": [{"text": instruccion}]}, {"role": "model", "parts": [{"text": "Entendido."}]}]
        for msj in datos.historial:
            role = "user" if msj["rol"] == "usuario" else "model"
            contents.append({"role": role, "parts": [{"text": msj["texto"]}]})
        contents.append({"role": "user", "parts": [{"text": datos.mensaje}]})
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=types.GenerateContentConfig(tools=[{"google_search": {}}])
        )
        return {"respuesta": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- ENDPOINTS OPERATIVOS Y MULTIMODALES ----------------

@app.post("/api/ocr-fiscal")
async def ocr_fiscal(archivos: List[UploadFile] = File(...)):
    try:
        resultados = []
        for archivo in archivos:
            file_bytes = await archivo.read()
            mime = reparar_mime(archivo.filename, archivo.content_type)
            prompt_ocr = """
            Paso 1: FILTRO DE CALIDAD. Si es ilegible devuelve EXACTAMENTE: {"error": "ERROR_CALIDAD_IMAGEN"}
            Paso 2: EXTRACCIÓN. Devuelve ÚNICAMENTE un JSON estricto: 
            {"rfc_emisor": "str", "rfc_receptor": "str", "subtotal": float, "total_traslados": float, "total_retenciones": float, "total": float, "conceptos": [{"descripcion": "str", "importe": float}]}
            """
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt_ocr]
            )
            
            try:
                datos = extraer_json(response.text)
                if "error" in datos:
                    resultados.append({"archivo": archivo.filename, "status": "error_calidad", "mensaje": "Imagen rechazada."})
                    continue
                
                sub = float(datos.get("subtotal", 0))
                tras = float(datos.get("total_traslados", 0))
                ret = float(datos.get("total_retenciones", 0))
                tot = float(datos.get("total", 0))
                calc = round(sub + tras - ret, 2)
                
                if abs(calc - tot) > 0.10:
                    resultados.append({"archivo": archivo.filename, "status": "error_matematico", "mensaje": f"Discrepancia. Matemáticas: {calc}, Documento: {tot}"})
                else:
                    resultados.append({"archivo": archivo.filename, "status": "success", "datos": datos})
            except:
                resultados.append({"archivo": archivo.filename, "status": "error", "mensaje": "Falla al estructurar datos."})
                
        if len(resultados) == 1:
            if resultados[0]["status"] == "success":
                return {"status": "success", "datos": resultados[0]["datos"]}
            else:
                return resultados[0]
                
        return {"status": "success_lote", "resultados_lote": resultados}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/risk-score-ia")
async def risk_score_ia(solicitud: SolicitudRiskScore):
    try:
        prompt = f"""
        Aplica Predictive Auditing basado en el modelo matemático Ra = Σ(wi * ki). 
        Puntaje actual: {solicitud.puntaje}/100. Factores detectados: {solicitud.factores}.
        Redacta un Dictamen de Auditoría Forense determinando el porcentaje exacto de probabilidad de recibir una Carta Invitación o facultades de comprobación del SAT (Art. 42 CFF) en los próximos 90 días basado en el índice de salud fiscal. Sé crudo y corporativo.
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2)
        )
        return {"analisis_legal": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/conciliacion-fuzzy")
async def conciliacion_fuzzy(datos: DatosConciliacion):
    try:
        prompt = """
        Eres un motor Fuzzy Matching. Relaciona cobros/pagos con facturas por semántica y temporalidad.
        Devuelve ÚNICAMENTE JSON VÁLIDO CON ESTA ESTRUCTURA EXACTA:
        {"conciliados": [{"id_banco": "str", "uuid_factura": "str", "razonamiento": "str", "nivel_confianza": int}]}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[prompt, json.dumps({"bancos": datos.bancos, "facturas": datos.facturas})],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
        )
        return extraer_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/materialidad")
async def validar_materialidad(contrato: UploadFile = File(...), datos_xml: str = Form(...)):
    try:
        file_bytes = await contrato.read()
        mime = reparar_mime(contrato.filename, contrato.content_type)
        prompt = f"""
        Auditor Forense (Art. 5-A). Cruza CONTRATO vs FACTURA XML: {datos_xml}.
        Devuelve ÚNICAMENTE un JSON válido:
        {{"nivel_riesgo": "BAJO"|"MEDIO"|"ALTO", "porcentaje_coincidencia": int, "dictamen_defensa": "str", "hallazgos": ["str"]}}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
        )
        return extraer_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tasa-efectiva")
async def evaluar_tasa_efectiva(datos: SolicitudTasaEfectiva):
    try:
        tasa = (datos.isr_pagado / datos.ingresos) * 100 if datos.ingresos > 0 else 0
        prompt = f"""
        Evalúa Tasa Efectiva ISR {tasa:.2f}% para el sector '{datos.sector}'.
        Compara contra parámetros del Plan Maestro de Fiscalización del SAT.
        Devuelve ÚNICAMENTE JSON:
        {{"tasa_calculada": {tasa:.2f}, "nivel_riesgo": "ALTO"|"MEDIO"|"BAJO", "dictamen_sectorial": "str"}}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return extraer_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auditoria-activos")
async def auditoria_activos(foto: UploadFile = File(...), datos_xml: str = Form(...)):
    try:
        file_bytes = await foto.read()
        mime = reparar_mime(foto.filename, foto.content_type)
        prompt = f"""
        Perito en materialidad fiscal. Analiza el activo en la FOTO y crúzalo contra el XML: {datos_xml}.
        Devuelve ÚNICAMENTE JSON: 
        {{"coincidencia": bool, "datos_extraidos_foto": "str", "dictamen_materialidad": "str"}}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
        )
        return extraer_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/precios-aduana")
async def precios_aduana(datos: SolicitudAduana):
    try:
        prompt = f"""
        Agente Aduanal ANAM. Conceptos XML: {json.dumps(datos.conceptos)}.
        Devuelve ÚNICAMENTE JSON: 
        {{"nivel_riesgo_aduanero": "ALTO"|"MEDIO"|"BAJO", "analisis_valoracion": "str"}}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return extraer_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prueba-servicio")
async def prueba_servicio(evidencia: UploadFile = File(...), datos_xml: str = Form(...)):
    try:
        file_bytes = await evidencia.read()
        mime = reparar_mime(evidencia.filename, evidencia.content_type)
        prompt = f"""
        Auditor SAT (Art. 69-B). Evalúa evidencia contra XML: {datos_xml}.
        Devuelve ÚNICAMENTE JSON: 
        {{"sustancia_economica_comprobada": bool, "nivel_riesgo_efos": "ALTO"|"MEDIO"|"BAJO", "dictamen_evidencia": "str"}}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return extraer_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/viaticos-geo")
async def viaticos_geo(datos_xml: str = Form(...)):
    try:
        prompt = f"""
        Auditor SAT (Art. 28 LISR). 
        Calcula distancia geoespacial y viáticos del XML: {datos_xml}.
        Devuelve ÚNICAMENTE JSON:
        {{"deducible": bool, "distancia_estimada_km": float, "limite_respetado": bool, "dictamen_viaticos": "str"}}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
        )
        return extraer_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/defensa-legal")
async def defensa_legal(documento: UploadFile = File(...)):
    try:
        file_bytes = await documento.read()
        mime = reparar_mime(documento.filename, documento.content_type)
        prompt = """
        Eres un abogado fiscalista senior. Redacta borrador de oficio para defender al contribuyente.
        NO incluyas marcas de agua ni asteriscos. Texto legal puro.
        """
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt],
            config=types.GenerateContentConfig(tools=[{"google_search": {}}], temperature=0.2)
        )
        return {"oficio_legal": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/banco-csv")
async def banco_csv(documento: UploadFile = File(...)):
    try:
        file_bytes = await documento.read()
        mime = reparar_mime(documento.filename, documento.content_type)
        prompt = """Extrae transacciones a CSV: Fecha (DD/MM/AAAA),Concepto,Cargo,Abono,Saldo"""
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt]
        )
        return {"csv_data": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analista-csf")
async def analista_csf(documento: UploadFile = File(...)):
    try:
        file_bytes = await documento.read()
        mime = reparar_mime(documento.filename, documento.content_type)
        
        prompt = """
        Analiza el documento fiscal.
        Devuelve ÚNICAMENTE JSON:
        {
            "rfc": "", "razon_social": "", "regimen_fiscal": [], "codigo_postal": "",
            "estatus_cumplimiento": "", "alertas": []
        }
        Reglas: estatus_cumplimiento es 'Estatus en padrón' (CSF) o 'Sentido de opinión' (32-D).
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
        )
        return extraer_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
