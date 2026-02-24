import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Tuple
import sys
from pathlib import Path
import tempfile
import os
import json
import re
import yaml
import requests
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "seg_crop_process"))
sys.path.insert(0, str(BASE / "mxfont_process"))
from mxfont_process.total_process_api import infer_from_path
from seg_crop_process.img_reconstruct import seg_crop
from fastapi.middleware.cors import CORSMiddleware

output_dir = Path("output_images")
output_dir.mkdir(parents=True, exist_ok=True)
UIUX_DIR = BASE / "uiux-app"
UI_DIST = UIUX_DIR / "dist"

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=10)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://0.0.0.0:5173",
        "http://192.168.0.10:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://0.0.0.0:5174",
        
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory="output_images"), name="outputs")

class Req(BaseModel):
    pairs: List[Tuple[str, str]]

CONFIG_PATH = BASE / "config.yaml"


def load_config():
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"config.yaml not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


CFG = load_config()


def get_project_dir(project_name: str):
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", project_name or "Default_Project")
    proj_path = output_dir / safe_name
    proj_path.mkdir(parents=True, exist_ok=True)
    return proj_path, safe_name


async def call_gemini_api(client, contents, config):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: client.models.generate_content(
            model=CFG.get("gemini", {}).get("model"),
            contents=contents,
            config=config,
        ),
    )

def detector(ref_img, tar_img):
    save_folder = "./temp"
    seg_result = seg_crop(ref_img, tar_img, save_folder)
    mx_weight = "./mxfont_process/weight/gen_2.pth"
    mx_cfg_path = "./mxfont_process/cfgs/defaults.yaml"
    seg_path = seg_result["output_filename"]
    if not os.path.isabs(seg_path):
        seg_path = str(Path(save_folder) / seg_path)
    print("##############get into detection##################")
    sim_result = infer_from_path(ref_img, seg_path, mx_weight, mx_cfg_path)
    return sim_result['style_sim'], sim_result['content_sim'], sim_result['style_pred'], sim_result['content_pred']

def _save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(upload.file.read())
        return f.name

@app.post("/api/detect")
def detect_api(
    ref_file: UploadFile = File(...),
    tar_file: UploadFile = File(...),
    project_name: str = Form(None),
):
    ref_path = _save_upload(ref_file)
    tar_path = _save_upload(tar_file)
    try:
        sim_s, sim_c, pred_s, pred_c = detector(ref_path, tar_path)
    finally:
        for path in (ref_path, tar_path):
            try:
                os.remove(path)
            except OSError:
                pass
    ok_s = "OK" in pred_s
    ok_c = "OK" in pred_c
    ok_count = int(ok_s) + int(ok_c)
    score = ((sim_s + sim_c) / 2) * 100
    verdict = "OK" if ok_count == 2 else "NG"
    return {
        "verdict": verdict,
        "score": score,
        "defects": [],
        "message": f"style={pred_s}, content={pred_c}",
    }


@app.post("/api/enhance")
def enhance_api(
    prompt: str = Form(...),
    options: str = Form(None),
    ref_images: List[UploadFile] = File(None),
):
    gpt_cfg = (CFG or {}).get("gpt", {})
    api_key = gpt_cfg.get("key")
    model = gpt_cfg.get("model") or "gpt-4.1-mini"
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing gpt.key in config.yaml")
    opts = {}
    if options:
        try:
            opts = json.loads(options)
        except json.JSONDecodeError:
            opts = {}
    lang = opts.get("lang", "en")
    length = opts.get("length", "medium")

    system_text = """You are a luxury beauty advertising prompt compiler optimized for Google Nano Banana Pro.
Take ANY user input prompt (any language) and output SIX (6) English prompts that reliably produce a high-end, photorealistic cosmetic campaign image.
If the user provides multiple reference product images (multiple cut-outs), treat them as a REQUIRED set.
Every output prompt must explicitly direct the generator to include ALL referenced products together in the same final image.

1) HARD OUTPUT TARGET (DO NOT NEGOTIATE)
- Aspect ratio MUST be: 2:3 portrait (vertical).
- Target render quality MUST be: true 4K look (include "4K" + "3840×5760" explicitly).
- The image MUST look like a real premium campaign photo (department-store / flagship e-commerce hero visual).

If the generator supports parameters, always set: aspect_ratio=2:3, resolution=3840x5760, quality=4k.
If it does not, you must still state these constraints in the prompt verbatim.

2) PRODUCT PRESERVATION (ABSOLUTE)
Never alter any provided product:
- No changes to silhouette, proportions, geometry, color, material, finish, transparency.
- No changes to any logo, typography, printed text, label layout, alignment, language, spacing, graphics.
- Products must remain clearly identifiable, fully visible, uncropped, and visually dominant as a set.

Label & typography behavior:
- Printed text/logo/label must behave physically and rotate with the product surface in perspective.
- Forbid "billboard text" where the product turns but the label stays front-facing.
- Text may be partially occluded due to angle/overlap, but must never be warped, stretched, mirrored, hallucinated, or redesigned.

Atmosphere may exist ONLY if it does not materially obscure any product.

3) MULTI-PRODUCT INCLUSION RULE (IF MULTIPLE REFERENCES PROVIDED)
- The final image MUST include ALL referenced products together in the same scene.
- Do NOT drop, replace, merge, duplicate, or invent products.
- Moderate overlap is allowed if all products remain clearly identifiable and mostly visible.

4) FREER ANGLE & COMPOSITION
All six prompts must preserve the same product set and core styling intent.
They may differ through more varied, editorial-leaning compositions, while staying physically plausible and premium.

Allowed angle freedom (expanded):
- Yaw: 0–60° per product or per group (including stronger three-quarter views).
- Camera elevation: from slightly below to moderately above (approx -5° to +12°), with natural perspective.
- Roll (tilt): mild dutch roll is allowed up to ~6° if it still feels like a luxury editorial ad (not chaotic).
- Lateral perspective shifts: allowed as long as there is no wide-angle distortion.

Allowed composition freedom (expanded):
- Placement can vary: centered, lower-third, left/right third, or gentle diagonal flow.
- Cropping must NEVER cut off any product; keep full visibility, but framing can be tighter or wider as long as premium.
- Negative space can be intentional: top-heavy, side-heavy, or asymmetric, as long as it looks designed.
- Layering and depth-stacking are allowed (front product + rear products) with controlled overlap.

Premium guardrails (must still hold):
- Use non-distorting lens language (avoid "wide-angle"); keep a premium product-photo look.
- Ensure products remain the clear subjects (no busy background).
- Maintain physical realism: correct shadows, reflections, contact points on the surface.
- Avoid chaotic perspective; keep it "editorial luxury," not "action shot."

Mandatory diversity across six prompts:
- At least two prompts must use stronger three-quarter yaw (>= 35°).
- At least one prompt may use a slightly lower camera angle (subtle hero perspective).
- At least one prompt may use mild dutch roll (<= 6°) with premium stability.
- At least two prompts must meaningfully change composition placement/negative space design.

5) CAMERA / LENS (REALISM FIRST)
- Premium product photography look; avoid wide-angle distortion.
- Medium-format feel preferred; if specifying focal length, keep it in the premium range (approx 50–110mm equivalent).
- Sharpness must feel real; do not oversharpen or create "crispy AI edges."
- Natural DOF: background separation allowed, but avoid heavy creamy bokeh unless it looks physically plausible.

6) LIGHTING (WHAT MAKES IT LOOK "REAL AD")
Use believable studio lighting (or luxury location-light if user asked for it):
- Soft diffused key light + controlled fill.
- Highlight control on glossy areas; no blown-out speculars over critical branding.
- Elegant shadow design for depth.
- Color palette: warm ivory / champagne / soft stone / muted neutrals unless user requests otherwise.

Avoid CG lighting: neon gradients, surreal glow, plastic reflections.

7) REQUIRED NEGATIVE CLAUSE (MUST BE INCLUDED IN EVERY PROMPT)
Explicitly forbid:
- CGI / 3D render look / illustration / painterly
- missing products / merged products / invented extra products
- billboard/front-facing label cheat when products are angled
- warped/mirrored/hallucinated text, logo distortion, brand redesign
- clutter, random props, messy scenes
- extreme dutch angles, fisheye, wide-angle distortion
- low-res, noise, artifacts, oversharpen halos, fake plastic lighting
- surreal objects, fantasy scenery, messy backgrounds

8) OUTPUT FORMAT (STRICT, MACHINE-PARSABLE)
- Output exactly six prompts in English, labeled exactly as:
  "Prompt 1:" ... "Prompt 6:"
- After EACH prompt, output a delimiter line exactly:
  "-----PROMPT-END-----"
- Do not output any other text, headers, commentary, JSON, bullets, numbering, or explanations.
- Each prompt must be 2–5 sentences, plain English.
- Each prompt must include: "2:3 portrait", "3840×5760", "4K".
- Each prompt must explicitly state:
  (a) that ALL referenced products must appear together in the same image (when multiple refs exist),
  (b) the chosen angle intent (yaw/elevation/roll described in natural language; exact degrees optional),
  (c) the typography-perspective rule (label rotates with the product; no billboard cheat).

READY-TO-PASTE BASE TEMPLATE (freer angle/composition; reuse; swap angle/composition details per prompt)
(Replace [PRODUCT SET] only; do not describe changes to any label/text.)

Create a photorealistic luxury beauty campaign photograph featuring ALL referenced products together as a single multi-product set ([PRODUCT SET]), rendered in 4K at 3840×5760 with a strict 2:3 vertical portrait composition. Use a more editorial, premium composition with intentional negative space and depth layering, choosing a distinct angle concept for this variant (e.g., stronger three-quarter yaw up to ~60°, slight low-angle hero perspective, or a mild dutch roll <= ~6°), while keeping every product fully visible and uncropped and never dropping, merging, duplicating, or inventing products. Ensure all printed labels behave physically and rotate in correct perspective with the product surfaces (no front-facing billboard label cheat); minor overlap is acceptable only if all products remain clearly identifiable and mostly visible, with realistic contact shadows and reflections on a refined studio surface. Light the scene with a soft diffused key and controlled fill against a minimal luxury background (ivory-to-champagne gradient or subtle architectural shadows) for clean highlights and premium depth. Avoid CGI/3D-render/illustration, missing/merged/invented products, warped/mirrored/hallucinated text, logo distortion, brand redesign, clutter, wide-angle distortion/fisheye, extreme tilt, low-resolution, noise, AI artifacts, surreal scenery, or busy backgrounds.
"""
    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_text}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ],
    }
    try:
        res = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if not res.ok:
        raise HTTPException(status_code=res.status_code, detail=res.text)
    data = res.json()
    text = (data.get("output_text") or "").strip()
    if not text:
        for out in data.get("output", []):
            for part in out.get("content", []):
                if part.get("type") in ("output_text", "text") and part.get("text"):
                    text = part["text"].strip()
                    break
            if text:
                break
    if not text:
        raise HTTPException(status_code=502, detail="Empty response from OpenAI")
    return {"enhanced_prompt": text}


@app.post("/api/generate")
async def generate_api(
    prompt: str = Form(...),
    project_name: str = Form("Default_Project"),
    ratio: str = Form("2:3"),
    size: str = Form("4096"),
    n: int = Form(1),
    ref_images: List[UploadFile] = File(None),
):
    gem_cfg = (CFG or {}).get("gemini", {})
    api_key = gem_cfg.get("key")
    model = gem_cfg.get("model")
    if not api_key:
        raise HTTPException(
            status_code=500, detail="Missing gemini.key in config.yaml"
        )
    if not model:
        raise HTTPException(
            status_code=500, detail="Missing gemini.model in config.yaml"
        )

    client = genai.Client(api_key=api_key)
    proj_path, safe_project_name = get_project_dir(project_name)

    contents = ["Reference Images: Use these for visual reference."]
    if ref_images:
        for img in ref_images:
            img_bytes = await img.read()
            contents.append(
                types.Part.from_bytes(
                    data=img_bytes, mime_type=img.content_type
                )
            )
    contents.append(prompt)

    image_config = types.ImageConfig(
        aspect_ratio=ratio,
        image_size="1K" if int(size) <= 1024 else "4K",
    )
    generate_config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=image_config,
    )

    tasks = [
        call_gemini_api(client, contents, generate_config)
        for _ in range(max(1, int(n)))
    ]
    responses = await asyncio.gather(*tasks)

    images = []
    for response in responses:
        file_name = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        out_path = proj_path / file_name
        for part in getattr(response, "parts", []):
            if getattr(part, "inline_data", None):
                img = part.as_image()
                img.save(out_path)
                images.append(
                    {
                        "url": f"http://127.0.0.1:8000/outputs/{safe_project_name}/{file_name}"
                    }
                )
                break

    if not images:
        raise HTTPException(
            status_code=500,
            detail="No image data returned from Gemini response.",
        )
    return {"images": images}


def ui_dist_ready() -> bool:
    return UI_DIST.is_dir() and (UI_DIST / "index.html").is_file()


@app.get("/")
def serve_root():
    if ui_dist_ready():
        return FileResponse(UI_DIST / "index.html")
    return HTMLResponse(
        "<h3>UI not built</h3>"
        "<p>Run <code>npm run build</code> in uiux-app, "
        "or use the dev server at http://localhost:5174</p>",
        status_code=200,
    )


@app.get("/{full_path:path}")
def serve_spa(full_path: str):
    if full_path.startswith(("api", "outputs")):
        raise HTTPException(status_code=404, detail="Not Found")
    if not ui_dist_ready():
        return HTMLResponse(
            "<h3>UI not built</h3>"
            "<p>Run <code>npm run build</code> in uiux-app, "
            "or use the dev server at http://localhost:5174</p>",
            status_code=200,
        )
    file_path = UI_DIST / full_path
    if file_path.is_file():
        return FileResponse(file_path)
    return FileResponse(UI_DIST / "index.html")
