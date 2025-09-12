from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import requests
import os
import time # Import the time library
import base64

# Optional local captioning (lazy loaded)
_blip_processor = None
_blip_model = None
_zs_pipeline = None

# Simple knowledge base for common plant diseases (leaf/fruit)
DISEASE_KB = {
    "powdery mildew": {
        "why": "Caused by fungal pathogens thriving in dry, warm days and cool, humid nights; poor air circulation.",
        "avoid": "Improve airflow, avoid overhead irrigation, prune crowded foliage, rotate crops, and use resistant varieties.",
        "treatment": "Apply sulfur or potassium bicarbonate fungicides; remove infected leaves; maintain spacing and sanitation.",
        "dosage": "Wettable sulfur 2–3 g/L water for foliar spray; potassium bicarbonate 5–10 g/L. Field: 1–1.5 kg/acre depending on canopy.",
    },
    "downy mildew": {
        "why": "Oomycete infection favored by cool, wet conditions and prolonged leaf wetness.",
        "avoid": "Water early so leaves dry quickly, increase spacing, rotate crops, use resistant varieties.",
        "treatment": "Copper-based fungicides at first sign; remove infected tissue; improve drainage and airflow.",
        "dosage": "Copper oxychloride 2–3 g/L water (200–300 g/100 L). Field: 1–2 kg/acre per spray, 7–10 day interval.",
    },
    "early blight": {
        "why": "Alternaria fungus; splashes from soil, high humidity, and plant stress.",
        "avoid": "Mulch to prevent soil splash, rotate crops 2–3 years, avoid overhead watering, fertilize adequately.",
        "treatment": "Copper or chlorothalonil fungicides; remove infected leaves; stake plants to improve airflow.",
        "dosage": "Chlorothalonil 2 g/L water; Mancozeb 2–2.5 g/L. Field: 1–1.5 kg/acre per application.",
    },
    "late blight": {
        "why": "Phytophthora infestans; cool, humid weather; spreads rapidly via spores.",
        "avoid": "Plant certified seed/seedlings, avoid overhead irrigation, destroy volunteers, rotate crops.",
        "treatment": "Immediate removal and destruction of infected plants; protectant fungicides containing mancozeb/cymoxanil where permitted.",
        "dosage": "Mancozeb 2–2.5 g/L; Cymoxanil + Mancozeb per label (commonly 1.5–2 g/L). Field: 1.5–2 kg/acre.",
    },
    "leaf spot": {
        "why": "Various fungi/bacteria; splash dispersal and high humidity.",
        "avoid": "Water at soil level, sanitize tools, remove debris, ensure spacing.",
        "treatment": "Copper sprays for bacterial spots; broad-spectrum fungicide for fungal spots; remove infected leaves.",
        "dosage": "Copper hydroxide 2 g/L; Captan 2 g/L for fungal spots. Field: ~1 kg/acre per spray.",
    },
    "rust": {
        "why": "Fungal rusts; spread by wind-borne spores in humid conditions.",
        "avoid": "Resistant cultivars, remove alternate hosts, avoid wet foliage.",
        "treatment": "Apply triazole or strobilurin fungicides per label; remove infected parts.",
        "dosage": "Propiconazole 1 ml/L water; Azoxystrobin 0.5 ml/L. Field: 200–300 ml/acre depending on formulation.",
    },
    "anthracnose": {
        "why": "Colletotrichum fungi causing fruit/leaf lesions; warm, wet weather.",
        "avoid": "Rotate crops, sanitize debris, improve airflow, avoid overhead watering.",
        "treatment": "Protectant fungicides; prune infected tissue; postharvest sanitation for fruits.",
        "dosage": "Carbendazim 1 g/L or Azoxystrobin 0.5 ml/L. Field: 200–300 ml or 0.5–1 kg/acre per label.",
    },
    "canker": {
        "why": "Fungal/bacterial pathogens entering wounds; stress and poor pruning practices.",
        "avoid": "Prune during dry weather, disinfect tools, avoid injuries, maintain vigor.",
        "treatment": "Prune 10–15 cm below lesions; dispose debris; copper sprays for bacterial cankers.",
        "dosage": "Copper oxychloride paste on wounds; spray 2–3 g/L after pruning. Field sprays as per label (~1–2 kg/acre).",
    },
    "mosaic virus": {
        "why": "Viral infection often vectored by aphids/whiteflies; transmitted by tools.",
        "avoid": "Control vectors, use virus-free seed, sanitize tools, remove weeds hosts.",
        "treatment": "No cure; rogue infected plants; manage vectors; plant resistant varieties.",
        "dosage": "For vectors: Neem oil 3–5 ml/L or Imidacloprid 0.3 ml/L as per local regulations.",
    },
    "nutrient deficiency": {
        "why": "Insufficient or imbalanced nutrients (N, P, K, Fe, Mg) and pH issues.",
        "avoid": "Soil test annually; maintain optimal pH; balanced fertilization; organic matter.",
        "treatment": "Apply specific nutrient amendments per soil test; foliar feeds for rapid correction.",
        "dosage": "General foliar feed: 1–2 g/L balanced NPK; Fe-EDDHA 0.5–1 g/L for iron chlorosis. Field: follow soil test recommendations.",
    },
    "sunscald": {
        "why": "High light/heat exposure damaging fruit/leaf tissues.",
        "avoid": "Provide shade cloth in heat waves; maintain foliage cover; avoid heavy pruning before hot days.",
        "treatment": "Remove damaged tissue if rotting; improve shading and irrigation scheduling.",
        "dosage": "Apply kaolin clay film 30–50 g/L as protective spray; irrigation 20–30 mm depending on soil moisture.",
    },
}

# Candidate labels for zero-shot image classification
ZS_LABELS = list(DISEASE_KB.keys()) + [
    "healthy leaf",
    "healthy fruit",
]

# --- App Initialization ---
app = FastAPI(
    title="PlantAI Diagnosis Backend (Hugging Face)",
    description="This API uses a Hugging Face model for a simple diagnosis with auto-retry logic.",
    version="2.2.0",
)

# --- Environment Variable for API Key ---
# IMPORTANT: You must set the HUGGING_FACE_TOKEN environment variable.
API_KEY = os.getenv("HUGGING_FACE_TOKEN")
# --- Inference API model endpoints to try (ordered by preference) ---
API_URLS = [
    "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base",
    "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large",
    "https://api-inference.huggingface.co/models/nlpconnect/vit-gpt2-image-captioning",
]

# --- CORS Configuration ---
origins = [
    "http://localhost",
    "http://localhost:8501",  # Default Streamlit port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Hugging Face Diagnosis Call with Auto-Retry ---
async def get_simple_diagnosis(image_bytes: bytes):
    """
    Calls a Hugging Face model with a retry mechanism to handle model loading.
    """
    print("Backend received a request. Preparing to call Hugging Face API...")
    if not API_KEY:
        print("ERROR: HUGGING_FACE_TOKEN is not set.")
        raise HTTPException(status_code=401, detail="Hugging Face API token is not configured. Please set the HUGGING_FACE_TOKEN environment variable.")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/octet-stream",
        "Accept": "application/json",
    }
    
    # Short-circuit to local captioning if requested
    if os.getenv("USE_LOCAL_ONLY", "").lower() in ("1", "true", "yes"): 
        print("USE_LOCAL_ONLY is set. Skipping remote inference and using local BLIP.")
        local_caption = local_image_caption(image_bytes)
        return _format_response_from_caption(local_caption, confidence=0.70)

    # --- Try multiple models with retry on 503/loading ---
    max_retries = 3
    last_error_message = None
    for api_url in API_URLS:
        print(f"Trying model endpoint: {api_url}")
        for attempt in range(max_retries):
            try:
                response = requests.post(api_url, headers=headers, data=image_bytes, timeout=60)

                if response.status_code == 200:
                    print("Successfully received a response from Hugging Face.")
                    result = response.json()
                    # Response can be a list of dicts with 'generated_text'
                    if isinstance(result, list) and result and isinstance(result[0], dict):
                        caption = result[0].get('generated_text') or result[0].get('caption') or 'Could not analyze image.'
                    elif isinstance(result, dict):
                        caption = result.get('generated_text') or result.get('caption') or 'Could not analyze image.'
                    else:
                        caption = 'Could not analyze image.'

                    return _format_response_from_caption(caption, confidence=0.90)

                if response.status_code == 503:
                    print(f"Attempt {attempt + 1}/{max_retries}: Model is loading, waiting 20 seconds before retrying...")
                    time.sleep(20)
                    continue

                # Non-200/503 → capture and try next endpoint
                last_error_message = f"{response.status_code} {response.text[:200]}" if response is not None else "Unknown error"
                print(f"Received an unexpected status code: {response.status_code} - trying next model if available")
                break

            except requests.exceptions.RequestException as e:
                last_error_message = str(e)
                print(f"An error occurred while communicating with Hugging Face: {e}")
                break

    # Fallback: return graceful response instead of 500 so the UI can proceed
    print("All model endpoints failed. Returning fallback analysis.")
    # Attempt local captioning before returning fallback
    try:
        print("Trying local BLIP captioning as a fallback...")
        local_caption = local_image_caption(image_bytes)
        return _format_response_from_caption(local_caption, confidence=0.70)
    except Exception as e:
        print(f"Local captioning failed: {e}")

    # Graceful final fallback
    return {
        "disease_name": "AI Analysis (Fallback)",
        "confidence": 0.10,
        "description": (
            "The captioning service is unavailable (" + (last_error_message or "unknown error") + "). "
            "Your image was received, but we could not generate a description right now."
        ),
        "treatment": (
            "Please try again later. If the issue persists, verify your HUGGING_FACE_TOKEN and network connectivity, "
            "or keep local mode enabled."
        ),
    }


def _format_response_from_caption(caption: str, confidence: float = 0.90):
    return {
        "disease_name": "AI Analysis",
        "confidence": confidence,
        "description": f"The AI model provided the following description: '{caption}'",
        "treatment": "Based on the description, please consult an agronomist for a specific treatment plan. This model provides a general analysis, not a detailed diagnosis.",
    }


def _ensure_local_blip_loaded():
    global _blip_processor, _blip_model
    if _blip_processor is not None and _blip_model is not None:
        return
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local captioning model could not be loaded: {e}")


def local_image_caption(image_bytes: bytes) -> str:
    _ensure_local_blip_loaded()
    try:
        from PIL import Image as PILImage
        import torch
        image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = _blip_processor(image, return_tensors="pt")
        with torch.no_grad():
            out = _blip_model.generate(**inputs, max_new_tokens=30)
        caption = _blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local captioning failed: {e}")


def _ensure_zero_shot_loaded():
    global _zs_pipeline
    if _zs_pipeline is not None:
        return
    try:
        from transformers import pipeline
        _zs_pipeline = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Zero-shot classifier could not be loaded: {e}")


def local_zero_shot_diagnose(image_bytes: bytes):
    _ensure_zero_shot_loaded()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        results = _zs_pipeline(image, candidate_labels=ZS_LABELS, hypothesis_template="a photo of {}")
        # results is list of dicts sorted by score
        top = results[0]
        label = top.get("label", "unknown").lower()
        score = float(top.get("score", 0.0))
        kb_key = label
        if kb_key not in DISEASE_KB:
            # map healthy
            if "healthy" in kb_key:
                return _format_knowledge_response(
                    name="Healthy",
                    confidence=score,
                    why="No disease indicators detected.",
                    avoid="Maintain current care; monitor regularly.",
                    treatment="No action needed.",
                )
            # fallback generic
            return _format_response_from_caption(f"Possible condition: {label}", confidence=score)

        info = DISEASE_KB[kb_key]
        return _format_knowledge_response(
            name=kb_key.title(),
            confidence=score,
            why=info["why"],
            avoid=info["avoid"],
            treatment=info["treatment"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local diagnosis failed: {e}")


# --- API Endpoint ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File provided is not a valid image.")

    try:
        image_bytes = await file.read()
        Image.open(io.BytesIO(image_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")
    
    # Try local zero-shot diagnosis first for actionable disease output
    try:
        zs_result = local_zero_shot_diagnose(image_bytes)
        return JSONResponse(content=zs_result)
    except Exception as e:
        print(f"Zero-shot diagnosis failed, falling back to captioning: {e}")

    diagnosis_result = await get_simple_diagnosis(image_bytes)
    return JSONResponse(content=diagnosis_result)

@app.get("/")
def read_root():
    return {"message": "Welcome to the PlantAI Backend (Hugging Face Version with Auto-Retry)."}


def _format_knowledge_response(name: str, confidence: float, why: str, avoid: str, treatment: str):
    # If the disease exists in KB and has dosage, append it to treatment
    dosage = None
    kb_key = name.lower()
    if kb_key in DISEASE_KB:
        dosage = DISEASE_KB[kb_key].get("dosage")
    treatment_text = treatment
    if dosage:
        treatment_text = f"{treatment}\n\nRecommended dosage: {dosage}"
    return {
        "disease_name": name,
        "confidence": confidence,
        "description": f"Why it occurs: {why}\n\nHow to avoid: {avoid}",
        "treatment": treatment_text,
    }

