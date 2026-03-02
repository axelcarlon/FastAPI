from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import os

app = FastAPI()

# Permite peticiones desde cualquier origen (Evita bloqueos durante pruebas)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("ADVERTENCIA: No se encontró GEMINI_API_KEY en el entorno.")

# Inicializar nuevo cliente de Gemini
client = genai.Client(api_key=api_key)

class SolicitudAnalisis(BaseModel):
    tipo_documento: str
    error_detectado: str
    descripcion_concepto: str

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

        # Uso del nuevo SDK de Google
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_sistema,
        )
        
        return {"analisis_legal": response.text.strip()}

    except Exception as e:
        print(f"Error interno: {str(e)}") # Esto se verá en los logs de Render si falla
        raise HTTPException(status_code=500, detail=f"Error en el procesamiento de IA: {str(e)}")

@app.get("/ping")
async def ping():
    return {"status": "ok"}
