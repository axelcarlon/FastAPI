from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import os

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
            contents=prompt_sistema,
        )
        return {"analisis_legal": response.text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_asesor(datos: MensajeChat):
    try:
        instruccion_sistema = """
        Eres el Asesor Fiscal Virtual de AuditorIA. Eres un experto en contabilidad mexicana, CFDI 4.0, Complementos de Pago (REP), Nómina 1.2, Ley del IVA, Ley del ISR y CFF.
        Tus respuestas deben ser directas, concretas, sin relleno, sin saludos ni despedidas. Nunca ejecutes cálculos matemáticos. Si te piden calcular un impuesto, indica la tasa o el fundamento legal, pero no el resultado aritmético.
        Comunícate de forma profesional, usando terminología contable mexicana (timbrado, deducción, EFOS, EDOS, PUE, PPD).
        """
        
        # Formatear el historial para Gemini
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

@app.get("/ping")
async def ping():
    return {"status": "ok"}
