import json, glob, os
from pathlib import Path

SRC = "/PROJECT/0990910002_A/dataset/NIA_raw_71748_extracted"
OUT = "/PROJECT/0990910002_A/dataset/NIA_final/univa_nia_corpus__v0.1.0.jsonl"

def norm(s: str) -> str:
    # 너무 공격적인 정규화는 피하고, 앞뒤 공백만 정리
    return s.strip()

count_files = 0
count_records = 0
count_written = 0

Path(os.path.dirname(OUT)).mkdir(parents=True, exist_ok=True)

with open(OUT, "w", encoding="utf-8") as out:
    for fp in sorted(glob.glob(SRC + "/**/*.json", recursive=True)):
        count_files += 1
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue

        data_info = obj.get("data_info", [])
        if not isinstance(data_info, list):
            continue

        for rec in data_info:
            count_records += 1
            if not isinstance(rec, dict):
                continue
            text = rec.get("contents", None)
            if not isinstance(text, str):
                continue
            text = norm(text)
            if not text:
                continue

            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count_written += 1

print(f"Done. files={count_files}, records={count_records}, written={count_written}")
print(f"Output: {OUT}")
