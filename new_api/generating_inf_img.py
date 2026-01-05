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
    """GPT를 사용하여 6개의 엄격한 형식 프롬프트를 생성합니다."""
    client = OpenAI(api_key=mykey)
    
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
    raw_segments = full_text.split("-----PROMPT-END-----")
    
    final_prompts = []
    for segment in raw_segments:
        clean_prompt = re.sub(r'Prompt \d[:.]', '', segment).strip()
        if len(clean_prompt) > 20:
            final_prompts.append(clean_prompt)
            
    return final_prompts[:6]

def call_gemini_multi_modal(mykey, ref_img_paths, prompt_text, out_name):
    """입력된 모든 이미지 경로를 동시에 사용하여 Gemini 이미지를 생성합니다."""
    client = genai.Client(api_key=mykey)
    
    contents = ["Reference Images (STRICT): Use these as the combined visual reference."]
    
    for img_path in ref_img_paths:
        p = Path(img_path)
        if not p.exists():
            print(f"⚠️ 경고: 파일을 찾을 수 없습니다: {img_path}")
            continue
            
        img_bytes = p.read_bytes()
        mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        img_part = types.Part.from_bytes(data=img_bytes, mime_type=mime)
        contents.append(img_part)
    
    contents.append(prompt_text)
    
    image_config = types.ImageConfig(
        aspect_ratio="2:3",
        image_size=info["gemini"].get("image_size", "1K"),
    )

    response = client.models.generate_content(
        model=info["gemini"]["model"],
        contents=contents,
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
    print("=== Luxury Ad Multi-Modal Generator (Direct Input Mode) ===")
    gpt_key = info["gpt"]["key"]
    gemini_key = info["gemini"]["key"]

    # 1. 텍스트 프롬프트 입력
    user_target = input("강화할 상품명이나 컨셉을 입력하세요: ")
    
    # 2. 이미지 파일명/경로 직접 입력
    print("\n[이미지 입력 가이드]")
    print("- '/content/ref_img' 폴더 내의 파일명(예: serum.png) 혹은 전체 경로를 입력하세요.")
    print("- 여러 장일 경우 쉼표(,)로 구분하세요 (1~5개 가능).")
    
    img_input = input("\n사용할 이미지 파일명/경로를 입력하세요: ")
    
    # 입력값 정제
    raw_paths = [x.strip() for x in img_input.split(",")]
    selected_img_paths = []
    
    base_dir = Path("/content/ref_img")
    for path in raw_paths:
        p = Path(path)
        # 파일명만 입력했을 경우를 대비해 기본 디렉토리와 결합 시도
        if not p.is_absolute() and not p.exists():
            p = base_dir / path
            
        if p.exists():
            selected_img_paths.append(str(p))
        else:
            print(f"❌ 파일을 찾을 수 없습니다: {path}")

    # 이미지 개수 검증
    if not (1 <= len(selected_img_paths) <= 5):
        print(f"❌ 유효한 이미지가 {len(selected_img_paths)}개입니다. 1~5개 사이로 입력해주세요.")
        return

    print(f"\n✅ 최종 선택된 이미지: {[Path(p).name for p in selected_img_paths]}")

    # [Step 1] GPT 변주 생성
    print(f"\n📝 '{user_target}'에 대해 6개의 변주 프롬프트를 생성 중...")
    enhanced_variants = call_gpt_six_variants(gpt_key, user_target)
    
    # [Step 2] CSV 저장
    variant_data = []
    for i, v_prompt in enumerate(enhanced_variants):
        variant_data.append({
            "variant_idx": i + 1,
            "used_images": ", ".join([Path(p).name for p in selected_img_paths]),
            "enhanced_prompt": v_prompt
        })
    
    prompt_df = pd.DataFrame(variant_data)
    prompt_df.to_csv("enhanced_prompts_direct_input.csv", index=False, encoding="utf-8-sig")
    print(f"✅ CSV 저장 완료: enhanced_prompts_direct_input.csv")

    # [Step 3] Gemini 호출
    print("\n🎨 이미지 생성 단계 (입력된 이미지들을 모두 참고합니다)...")
    safe_name = re.sub(r'[^\w\s-]', '', user_target).strip().replace(' ', '_')
    
    for idx, row in prompt_df.iterrows():
        v_idx = row['variant_idx']
        p_text = row['enhanced_prompt']
        file_name = f"{safe_name}_V{v_idx}"
        
        print(f"[{v_idx}/6] 생성 중...")
        try:
            call_gemini_multi_modal(gemini_key, selected_img_paths, p_text, file_name)
        except Exception as e:
            print(f"   ⚠️ 오류: {e}")

    print("\n✨ 모든 작업이 완료되었습니다!")

if __name__ == "__main__":
    main()
