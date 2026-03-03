from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import json
import os
import re

app = FastAPI()

# Configuración CORS para permitir peticiones desde tu frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicialización del cliente de Gemini
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


# ---------------- ENDPOINTS ----------------

@app.get("/ping")
async def ping():
    return {"status": "ok", "mensaje": "Servidor AuditorIA Activo"}


@app.post("/api/asesor-ia")
async def analizar_discrepancia(solicitud: SolicitudAnalisis):
    try:
        prompt_sistema = f"""
        Eres el motor de Inteligencia Artificial de AuditorIA, experto en la Ley del ISR, Ley del IVA y Código Fiscal de la Federación (CFF) de México.
        REGLA INQUEBRANTABLE: Nunca realices operaciones matemáticas. Tu único trabajo es justificar legalmente el error que se te proporciona.
        
        Analiza el siguiente error detectado por nuestro sistema matemático:
        - Tipo de CFDI: {solicitud.tipo_documento}
        - Concepto facturado: {solicitud.descripcion_concepto}
        - Error detectado: {solicitud.error_detectado}
        
        Devuelve una respuesta breve, directa y profesional (máximo 4 renglones). 
        Cita el artículo de la ley aplicable (ISR, IVA o CFF) que justifica por qué el error detectado es una discrepancia fiscal y qué multa podría aplicar.
        No uses saludos ni despedidas. Sé frío y analítico.
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt_sistema
        )
        return {"analisis_legal": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_asesor(datos: MensajeChat):
    try:
        instruccion_sistema = """
        Eres el Asesor Fiscal Virtual de AuditorIA (te llamas Paula). Eres un experto en contabilidad mexicana, CFDI 4.0, Complementos de Pago (REP), Nómina 1.2, Ley del IVA, Ley del ISR y CFF.
        Tus respuestas deben ser directas, concretas, sin relleno, sin saludos ni despedidas. Nunca ejecutes cálculos matemáticos. Si te piden calcular un impuesto, indica la tasa o el fundamento legal, pero no el resultado aritmético.
        Comunícate de forma profesional, usando terminología contable mexicana (timbrado, deducción, EFOS, EDOS, PUE, PPD).
        """
        
        contents = [{"role": "user", "parts": [{"text": instruccion_sistema}]}]
        contents.append({"role": "model", "parts": [{"text": "Entendido. Operaré bajo estos parámetros."}]})
        
        for msj in datos.historial:
            role = "user" if msj["rol"] == "usuario" else "model"
            contents.append({"role": role, "parts": [{"text": msj["texto"]}]})
            
        contents.append({"role": "user", "parts": [{"text": datos.mensaje}]})

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents
        )
        return {"respuesta": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ocr-fiscal")
async def ocr_fiscal(archivo: UploadFile = File(...)):
    try:
        file_bytes = await archivo.read()
        mime_type = archivo.content_type

        prompt_ocr = """
        Actúa como un auditor fiscal automatizado. Analiza el documento adjunto (factura o recibo).
        
        Paso 1: FILTRO DE CALIDAD. Si la imagen está borrosa, ilegible, incompleta o no parece un comprobante de pago/factura válido, detente inmediatamente y devuelve EXACTAMENTE Y ÚNICAMENTE este JSON:
        {"error": "ERROR_CALIDAD_IMAGEN"}
        
        Paso 2: EXTRACCIÓN. Si la imagen es legible, extrae los siguientes datos y devuelve ÚNICAMENTE un objeto JSON con esta estructura exacta, sin texto adicional ni formato markdown (sin bloques ```json):
        {
            "rfc_emisor": "string",
            "rfc_receptor": "string",
            "subtotal": float,
            "total_traslados": float,
            "total_retenciones": float,
            "total": float,
            "conceptos": [
                {"descripcion": "string", "importe": float}
            ]
        }
        Si no encuentras retenciones o traslados, asigna 0.0. Asegúrate de extraer las cifras numéricas puras.
        """

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                prompt_ocr
            ]
        )

        texto_limpio = re.sub(r'```json\n|```', '', response.text).strip()
        
        try:
            datos_extraidos = json.loads(texto_limpio)
        except json.JSONDecodeError:
            return {"status": "error", "mensaje": "La IA no pudo estructurar los datos correctamente. Intenta con una imagen más clara."}

        if "error" in datos_extraidos and datos_extraidos["error"] == "ERROR_CALIDAD_IMAGEN":
            return {"status": "error_calidad", "mensaje": "Imagen rechazada por baja calidad, borrosa o incompleta."}

        # BARRERA MATEMÁTICA
        subtotal = float(datos_extraidos.get("subtotal", 0.0))
        traslados = float(datos_extraidos.get("total_traslados", 0.0))
        retenciones = float(datos_extraidos.get("total_retenciones", 0.0))
        total_declarado = float(datos_extraidos.get("total", 0.0))

        calculo_matematico = round(subtotal + traslados - retenciones, 2)
        
        if abs(calculo_matematico - total_declarado) > 0.10:
            return {
                "status": "error_matematico", 
                "mensaje": f"Discrepancia detectada. Matemáticas: ${calculo_matematico}. Documento marca: ${total_declarado}.",
                "datos_extraidos": datos_extraidos
            }

        return {"status": "success", "datos": datos_extraidos}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/risk-score-ia")
async def risk_score_ia(solicitud: SolicitudRiskScore):
    try:
        prompt = f"""
        Eres el Auditor Fiscal Jefe de AuditorIA. Se ha procesado un lote masivo de XMLs.
        El sistema ha calculado un 'RiskScore' (0 a 100, donde 100 es perfecto cumplimiento y < 70 es riesgo inminente de auditoría).
        
        Puntaje obtenido: {solicitud.puntaje}/100
        Factores de riesgo detectados en el lote: {solicitud.factores}
        
        Redacta un 'Dictamen de Auditoría Forense' breve (máximo 5 renglones). 
        Indica la probabilidad de que el SAT inicie facultades de comprobación (Art. 42 CFF) o suspenda sellos digitales (Art. 17-H Bis). 
        Menciona las multas aplicables según el CFF basadas en los factores detectados. Sé directo, frío y corporativo.
        """
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=prompt
        )
        return {"analisis_legal": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/conciliacion-fuzzy")
async def conciliacion_fuzzy(datos: DatosConciliacion):
    try:
        prompt = """
        Eres un agente experto en auditoría y conciliación bancaria mexicana.
        Recibirás dos listas en formato JSON: 'bancos' (movimientos sin identificar) y 'facturas' (CFDIs pendientes de cobro/pago).
        Tu tarea es aplicar 'Fuzzy Matching' semántico para encontrar qué pago corresponde a qué factura.
        Razona basándote en la similitud de los montos, la proximidad de fechas y el análisis semántico del concepto bancario frente al nombre/RFC del emisor/receptor o la descripción de la factura.
        
        Devuelve ÚNICAMENTE un JSON con esta estructura exacta, sin texto adicional ni bloques markdown (```json):
        {
            "conciliados": [
                {
                    "id_banco": "string",
                    "uuid_factura": "string",
                    "razonamiento": "Explicación breve de por qué coinciden",
                    "nivel_confianza": 0 a 100
                }
            ]
        }
        Solo incluye los que tengan un nivel de confianza mayor a 60.
        """
        
        contenido = {
            "bancos": datos.bancos,
            "facturas": datos.facturas
        }
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, json.dumps(contenido)]
        )
        
        texto_limpio = re.sub(r'```json\n|```', '', response.text).strip()
        return json.loads(texto_limpio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
