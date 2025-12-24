from pathlib import Path
from openai import OpenAI
import yaml

from google import genai
from google.genai import types
import pandas as pd

# 지침 불러오기(.txt)
guide_path = Path(__file__).resolve().parent / "prompting_guide.txt"
system_instructions = guide_path.read_text(encoding="utf-8")
cfg_path = Path("config.yaml")
info = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

def save_images(response, name: str, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    idx = 0

    # 응답의 part들 중 inline_data가 이미지입니다. :contentReference[oaicite:1]{index=1}
    def handle_parts(parts):
        nonlocal idx
        for part in parts or []:
            if getattr(part, "inline_data", None):
                img = part.as_image()
                out_path = out_dir / f"{name}_tgt{idx:02d}.png"  # 저장할 생성 이미지 이름
                img.save(out_path)
                saved.append(out_path)
                idx += 1

    candidates = getattr(response, "candidates", None) or []
    print("batch :",candidates)
    if candidates:
        for cand in candidates:
            content = getattr(cand, "content", None)
            handle_parts(getattr(content, "parts", None))
    else:
        handle_parts(getattr(response, "parts", None))

    # 텍스트만 오는 경우(response에 이미지 없을 때)
    if not saved and getattr(response, "text", None):
        (out_dir / f"{name}_response_text.txt").write_text(response.text, encoding="utf-8")

    return saved

def call_gpt(mykey: str, user_prompt: str) -> str:
    client = OpenAI(api_key=mykey)
    final_user_content = (
        f"{user_prompt}\n\n"
        "이제 사용자가 입력한 거에서 내 지침에 맞는 강화된 프롬프트를 생성해줘."
    )
    response = client.responses.create(
        model=info["gpt"]["model"],
        temperature = 0.8,
        input=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": final_user_content},
        ],
    )
    return response.output_text

def call_gemini(mykey, ref_img, prompt):
    client = genai.Client(api_key=mykey)

    # 이미지 파일을 바이트로 읽어 Part로 감싸기
    img_bytes = Path(ref_img).read_bytes()
    img_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
    # input
    contents = [
        "Reference Image (STRICT): Use this as the visual reference.",
        img_part,
        prompt,
    ]
    # gemini 호출
    response = client.models.generate_content(
        model=info["gemini"]["model"],
        temperature = 0.8,
        contents=contents,
        config=types.GenerateContentConfig(
            candidate_count=info["gemini"]["batch"],
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=info["gemini"]["aspect_ratio"],
                image_size=info["gemini"]["image_size"],
            ),
        ),
    )

    out_dir = Path(info["output"]["output_dir"])
    img_file_name = Path(ref_img).stem
    saved = save_images(response, img_file_name, out_dir) # out_dir / img_file_name
    return saved

def main():
    gpt_key = info["gpt"]["key"]
    df = pd.read_csv("data.csv") # 파일 받기
    # 프롬프트 리스트 파일 불러오기
    user_prompt = df["col1"].tolist()
    # 레퍼런스 리스트 파일 불러오기
    input_ref = df["col2"].tolist()
    user_prompt = list() # 사용자 프롬프트.
    input_ref = list() # 여기에 넣을 reference 이미지 경로
    for user in user_prompt:
        for input_img in input_ref:
            # 프롬프트 생성(gemini에 넣을 프롬프트 생성)
            gen_prompt = call_gpt(gpt_key, user)
            print("📷reference image📷")
            print(input_img)
            print("✅생성된 프롬프트✅")
            print(gen_prompt)
            print()

            # 제미나이 이미지 생성
            gemini_key = info["gemini"]["key"]
            saved_img = call_gemini(gemini_key, input_img, gen_prompt)

            print(f'🤖Model: {info["gemini"]["model"]}')
            print(f'📊Output: {Path(info["output"]["output_dir"]).resolve()}')
            for p in saved_img:
                print(f"💾Saved:  {p.resolve()}")
            print("--------------------------------------------------")

if __name__ == "__main__":
    main()
