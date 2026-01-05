from pathlib import Path
from openai import OpenAI
import yaml
from google import genai
from google.genai import types
import pandas as pd
import re

# 1. 환경 설정 및 지침 로드
guide_path = Path(__file__).resolve().parent / "prompting_guide_inf.txt"
system_instructions = guide_path.read_text(encoding="utf-8")
cfg_path = Path("config.yaml")
info = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

def call_gpt_six_variants(mykey: str, user_input: str) -> list[str]:
    """
    GPT를 호출하여 엄격한 형식(Prompt X: ... -----PROMPT-END-----)의 
    6개 변주 프롬프트를 리스트로 반환합니다.
    """
    client = OpenAI(api_key=mykey)
    
    # 지침에 출력 형식(Strict Format) 강제 추가
    format_instruction = (
        "\n\nOUTPUT FORMAT (STRICT, MACHINE-PARSABLE):\n"
        "- Output exactly six prompts in English, labeled exactly as: 'Prompt 1:' ... 'Prompt 6:'\n"
        "- After EACH prompt, output a delimiter line exactly: '-----PROMPT-END-----'\n"
        "- Do not output any other text, headers, commentary, JSON, or explanations."
    )
    
    response = client.chat.completions.create(
        model=info["gpt"]["model"],
        temperature=0.8,
        messages=[
            {"role": "system", "content": system_instructions + format_instruction},
            {"role": "user", "content": f"Target Product/Concept: {user_input}"},
        ],
    )
    full_text = response.choices[0].message.content
    
    # 1. -----PROMPT-END----- 구분자로 먼저 분할
    raw_segments = full_text.split("-----PROMPT-END-----")
    
    final_prompts = []
    for segment in raw_segments:
        # 2. "Prompt X:" 레이블 제거 및 앞뒤 공백 정리
        clean_prompt = re.sub(r'Prompt \d[:.]', '', segment).strip()
        if len(clean_prompt) > 20: # 유효한 길이의 프롬프트만 추가
            final_prompts.append(clean_prompt)
            
    return final_prompts[:6]

def call_gemini_nano(mykey, ref_img_path, prompt_text, out_name):
    """지침에 명시된 2:3 비율로 Gemini 이미지를 생성합니다."""
    client = genai.Client(api_key=mykey)
    img_bytes = Path(ref_img_path).read_bytes()
    img_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
    
    image_config = types.ImageConfig(
        aspect_ratio="2:3",
        image_size=info["gemini"].get("image_size", "1K"),
    )

    response = client.models.generate_content(
        model=info["gemini"]["model"],
        contents=["Reference Image (STRICT):", img_part, prompt_text],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=image_config,
        ),
    )

    out_dir = Path(info["output"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    for part in getattr(response, "parts", []):
        if getattr(part, "inline_data", None):
            img = part.as_image()
            out_path = out_dir / f"{out_name}.png"
            img.save(out_path)
            saved_files.append(out_path)
    return saved_files

def main():
    print("=== Luxury Ad Variant Generator (Strict Format Mode) ===")
    gpt_key = info["gpt"]["key"]
    gemini_key = info["gemini"]["key"]

    user_target = input("강화할 상품명이나 컨셉을 입력하세요: ")
    
    img_dir = Path("/content/ref_img")
    ref_images = [str(f) for f in img_dir.glob("*") if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    
    if not ref_images:
        print("❌ 레퍼런스 이미지를 찾을 수 없습니다.")
        return
    
    ref_img = ref_images[0]

    # [Step 1] GPT 변주 생성
    print(f"\n📝 '{user_target}'에 대해 6개의 변주 프롬프트를 생성 중...")
    enhanced_variants = call_gpt_six_variants(gpt_key, user_target)
    
    if len(enhanced_variants) < 6:
        print(f"⚠️ 경고: 프롬프트가 {len(enhanced_variants)}개만 추출되었습니다. 형식을 확인하세요.")

    # [Step 2] CSV 저장
    variant_data = []
    for i, v_prompt in enumerate(enhanced_variants):
        variant_data.append({
            "variant_idx": i + 1,
            "enhanced_prompt": v_prompt
        })
    
    prompt_df = pd.DataFrame(variant_data)
    prompt_df.to_csv("enhanced_prompts_list.csv", index=False, encoding="utf-8-sig")
    print(f"✅ CSV 저장 완료: enhanced_prompts_list.csv")

    # [Step 3] Gemini 호출
    print("\n🎨 이미지 생성 단계 (나노 바나나 6회 호출)...")
    safe_name = re.sub(r'[^\w\s-]', '', user_target).strip().replace(' ', '_')
    
    for idx, row in prompt_df.iterrows():
        v_idx = row['variant_idx']
        p_text = row['enhanced_prompt']
        file_name = f"{safe_name}_V{v_idx}"
        
        print(f"[{v_idx}/6] 생성 중...")
        try:
            call_gemini_nano(gemini_key, ref_img, p_text, file_name)
        except Exception as e:
            print(f"   ⚠️ 오류: {e}")

    print("\n✨ 모든 작업이 완료되었습니다!")

if __name__ == "__main__":
    main()
