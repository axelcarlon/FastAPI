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

def extraer_json(texto: str):
    texto = re.sub(r'```json\n|```', '', texto).strip()
    return json.loads(texto)

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

@app.post("/api/ocr-fiscal")
async def ocr_fiscal(archivos: List[UploadFile] = File(...)):
    try:
        resultados = []
        for archivo in archivos:
            file_bytes = await archivo.read()
            mime_type = archivo.content_type
            prompt_ocr = """
            Paso 1: FILTRO DE CALIDAD. Si es ilegible devuelve EXACTAMENTE: {"error": "ERROR_CALIDAD_IMAGEN"}
            Paso 2: EXTRACCIÓN. Devuelve ÚNICAMENTE un JSON estricto: 
            {"rfc_emisor": "str", "rfc_receptor": "str", "subtotal": float, "total_traslados": float, "total_retenciones": float, "total": float, "conceptos": [{"descripcion": "str", "importe": float}]}
            """
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime_type), prompt_ocr]
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
        Devuelve ÚNICAMENTE JSON VÁLIDO CON ESTA ESTRUCTURA EXACTA, sin texto adicional:
        {"conciliados": [{"id_banco": "str", "uuid_factura": "str", "razonamiento": "str", "nivel_confianza": int}]}
        Solo incluye coincidencias con nivel de confianza > 60.
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[prompt, json.dumps({"bancos": datos.bancos, "facturas": datos.facturas})],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/materialidad")
async def validar_materialidad(contrato: UploadFile = File(...), datos_xml: str = Form(...)):
    try:
        file_bytes = await contrato.read()
        mime_type = contrato.content_type
        prompt = f"""
        Auditor Forense (Art. 5-A). Cruza CONTRATO vs FACTURA XML: {datos_xml}.
        REGLA ESTRICTA: Si el monto o la descripción del servicio estipulado en el contrato difiere sustancialmente del XML facturado, clasifica el riesgo como "ALTO" automáticamente por simulación de operaciones.
        Devuelve ÚNICAMENTE un JSON válido con la siguiente estructura exacta:
        {{"nivel_riesgo": "BAJO"|"MEDIO"|"ALTO", "porcentaje_coincidencia": int, "dictamen_defensa": "str", "hallazgos": ["str"]}}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
        )
        return json.loads(response.text)
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
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/auditoria-activos")
async def auditoria_activos(foto: UploadFile = File(...), datos_xml: str = Form(...)):
    try:
        file_bytes = await foto.read()
        mime_type = foto.content_type
        prompt = f"""
        Perito en materialidad fiscal. Analiza el activo en la FOTO y crúzalo contra el XML: {datos_xml}.
        REGLA DE MERCADO: Evalúa estrictamente la razonabilidad económica. Si el objeto en la foto es de bajo valor (ej. un foco, una pluma) y el XML indica un monto absurdo (ej. $50,000), o si el XML factura "Servicios" pero la foto es un bien físico, MÁRCALO COMO DISCREPANCIA (coincidencia: false).
        Devuelve ÚNICAMENTE JSON: 
        {{"coincidencia": bool, "datos_extraidos_foto": "str", "dictamen_materialidad": "str"}}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/precios-aduana")
async def precios_aduana(datos: SolicitudAduana):
    try:
        prompt = f"""
        Agente Aduanal ANAM. Conceptos XML a evaluar: {json.dumps(datos.conceptos)}.
        BARRERA DE ENTRADA: Si los conceptos no corresponden a importación/exportación o no tienen complementos de comercio exterior evidentes, el riesgo es "BAJO" y el dictamen debe decir "Este XML no presenta características de operaciones aduaneras o de importación directa."
        Si son de aduana, indica si existe subvaluación en valores unitarios frente a referencias internacionales.
        Devuelve ÚNICAMENTE JSON: 
        {{"nivel_riesgo_aduanero": "ALTO"|"MEDIO"|"BAJO", "analisis_valoracion": "str"}}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prueba-servicio")
async def prueba_servicio(evidencia: UploadFile = File(...), datos_xml: str = Form(...)):
    try:
        file_bytes = await evidencia.read()
        mime_type = evidencia.content_type
        prompt = f"""
        Auditor SAT (Art. 69-B). Evalúa evidencia adjunta contra XML: {datos_xml}.
        Verifica congruencia y sustancia económica. 
        Devuelve ÚNICAMENTE JSON: 
        {{"sustancia_economica_comprobada": bool, "nivel_riesgo_efos": "ALTO"|"MEDIO"|"BAJO", "dictamen_evidencia": "str"}}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime_type), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/viaticos-geo")
async def viaticos_geo(datos_xml: str = Form(...)):
    try:
        prompt = f"""
        Auditor SAT (Art. 28 LISR, Fracción V). 
        Calcula la distancia geoespacial entre el Código Postal del Emisor y el Código Postal del Receptor indicados en el siguiente XML: {datos_xml}.
        REGLA: Los viáticos y gastos de viaje solo son deducibles si se erogan a una distancia de la faja de 50 kilómetros que circunde al establecimiento del contribuyente.
        Valida además los topes diarios de alimentación ($750 MXN nacional, $1500 extranjero).
        Devuelve ÚNICAMENTE JSON válido:
        {{"deducible": bool, "distancia_estimada_km": float, "limite_respetado": bool, "dictamen_viaticos": "str"}}
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
        )
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/defensa-legal")
async def defensa_legal(documento: UploadFile = File(...)):
    try:
        file_bytes = await documento.read()
        mime_type = documento.content_type
        prompt = """
        Eres un abogado fiscalista senior. El documento adjunto es un requerimiento, multa o Carta Invitación del SAT.
        Redacta el borrador del oficio de aclaración o recurso de revocación formal para defender al contribuyente.
        REGLA DE FORMATO INQUEBRANTABLE: Redacta un documento PROFESIONAL, crudo y directo. 
        NO incluyas marcas de agua, ni texto como 'Generado por IA', ni asteriscos de markdown. Solo el texto legal puro listo para firma.
        """
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime_type), prompt],
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.2
            )
        )
        return {"oficio_legal": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/banco-csv")
async def banco_csv(documento: UploadFile = File(...)):
    try:
        file_bytes = await documento.read()
        mime_type = documento.content_type
        prompt = """
        Eres un extractor financiero. Analiza el estado de cuenta bancario adjunto (PDF o Imagen).
        Extrae todas las transacciones y devuélvelas ÚNICAMENTE en formato CSV con las siguientes columnas exactas:
        Fecha (DD/MM/AAAA),Concepto,Cargo,Abono,Saldo
        No incluyas texto adicional, ni saludos, ni bloques markdown. Solo el texto CSV puro.
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime_type), prompt]
        )
        return {"csv_data": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analista-csf")
async def analista_csf(documento: UploadFile = File(...)):
    try:
        file_bytes = await documento.read()
        mime = documento.content_type
        if not mime or mime == "application/octet-stream":
            mime = "application/pdf"
        
        prompt = """
        Analiza el documento fiscal.
        Devuelve la información estrictamente en el siguiente formato JSON:
        {
            "rfc": "",
            "razon_social": "",
            "regimen_fiscal": [],
            "codigo_postal": "",
            "estatus_cumplimiento": "",
            "alertas": []
        }
        Reglas de extracción:
        - estatus_cumplimiento: Si el documento es una Constancia de Situación Fiscal, escribe el 'Estatus en el padrón' (ej. ACTIVO). Si es una Opinión de Cumplimiento 32-D, escribe el sentido de la opinión (ej. POSITIVA o NEGATIVA).
        - alertas: Si no hay irregularidades, devuelve un arreglo vacío.
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime), prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
        )
        return extraer_json(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
