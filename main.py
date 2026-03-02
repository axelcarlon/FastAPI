from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os

# Inicializar aplicación
app = FastAPI()

# Configurar CORS (Permite que tu dominio de frontend se comunique con este servidor)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://auditoriaxml.mx", "http://localhost:3000"], # Restringe a tu dominio en prod
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# Configurar API Key de Gemini desde variables de entorno (Configúralo en Render)
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("ADVERTENCIA: No se encontró GEMINI_API_KEY en el entorno.")
genai.configure(api_key=api_key)

# Modelo de datos que esperamos recibir del Frontend
class SolicitudAnalisis(BaseModel):
    tipo_documento: str # Ej: "Factura PPD", "Nomina"
    error_detectado: str # Ej: "Retencion ISR calculada al 10%, debió ser 1.25%"
    descripcion_concepto: str # Ej: "Servicios profesionales de fletes"

@app.post("/api/asesor-ia")
async def analizar_discrepancia(solicitud: SolicitudAnalisis):
    try:
        # Prompt maestro que condiciona a la IA a comportarse como auditor fiscal
        prompt_sistema = """
        Eres el motor de Inteligencia Artificial de AuditorIA, experto en la Ley del ISR, Ley del IVA y Código Fiscal de la Federación (CFF) de México.
        REGLA INQUEBRANTABLE: Nunca realices operaciones matemáticas. Tu único trabajo es justificar legalmente el error que se te proporciona.
        
        Analiza el siguiente error detectado por nuestro sistema matemático:
        - Tipo de CFDI: {tipo_documento}
        - Concepto facturado: {descripcion_concepto}
        - Error detectado: {error_detectado}
        
        Devuelve una respuesta breve, directa y profesional (máximo 4 renglones). 
        Cita el artículo de la ley aplicable (ISR, IVA o CFF) que justifica por qué el error detectado es una discrepancia fiscal y qué multa podría aplicar.
        No uses saludos ni despedidas. Sé frío y analítico.
        """.format(
            tipo_documento=solicitud.tipo_documento,
            descripcion_concepto=solicitud.descripcion_concepto,
            error_detectado=solicitud.error_detectado
        )

        # Configurar modelo Gemini 1.5 Pro
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
        # Generar respuesta
        response = model.generate_content(prompt_sistema)
        
        return {"analisis_legal": response.text.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento de IA: {str(e)}")

# Punto de control para mantener vivo el servidor en Render
@app.get("/ping")
async def ping():
    return {"status": "ok"}