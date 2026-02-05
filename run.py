# run.py
import os
import json
from dotenv import load_dotenv
import pandas as pd

from prompts import (
    SYSTEM_PROMPT_OPEN,
    SYSTEM_PROMPT_FILTER,
    SYSTEM_PROMPT_AXIAL,
    SYSTEM_PROMPT_SELECTIVE,
    SYSTEM_PROMPT_STORYLINE,
)

# === 按你的 Module 命名导入 ===
from Module1_open_coding import run_open_coding_from_text
from Module2_filtering import filter_unique_codes_with_batching, build_filter_outputs_from_open_df
from Module3_axial_coding import (
    deepseek_axial_coding,
    attach_axial_to_retain_unique,
    make_axial_summary,
    attach_axial_to_row_level,
)
from Module4_selective_coding import deepseek_selective_coding, validate_coverage
from Module5_storyline import generate_storyline, validate_storyline_result


def must_get_env(key: str) -> str:
    v = os.getenv(key, "").strip()
    if not v:
        raise RuntimeError(f"Missing env var: {key}")
    return v


def ensure_dirs(*dirs: str):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def main():
    load_dotenv()

    # =====================================================
    # 0) Env Config
    # =====================================================
    api_key = must_get_env("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions").strip()

    model_open = os.getenv("DEEPSEEK_OPEN_MODEL", "deepseek-chat").strip()
    model_filter = os.getenv("DEEPSEEK_FILTER_MODEL", "deepseek-reasoner").strip()
    model_axial = os.getenv("DEEPSEEK_AXIAL_MODEL", "deepseek-reasoner").strip()
    model_reasoner = os.getenv("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner").strip()

    # =====================================================
    # 1) Paths (Relative)
    # =====================================================
    input_txt = "data/interview_merged.txt"

    out_root = "outputs"
    out_filter_dir = os.path.join(out_root, "filtering")
    out_axial_dir = os.path.join(out_root, "axial")
    out_selective_dir = os.path.join(out_root, "selective")
    out_story_dir = os.path.join(out_root, "storyline")

    ensure_dirs(out_root, out_filter_dir, out_axial_dir, out_selective_dir, out_story_dir)

    # =====================================================
    # Step 1) Open Coding
    # =====================================================
    with open(input_txt, "r", encoding="utf-8") as f:
        text = f.read()

    open_df = run_open_coding_from_text(
        text=text,
        base_url=base_url,
        api_key=api_key,
        model=model_open,
        system_prompt=SYSTEM_PROMPT_OPEN,
        sleep_time=1.0,
    )

    open_xlsx = os.path.join(out_root, "open_coding.xlsx")
    open_df.to_excel(open_xlsx, index=False)
    print("✅ [Module1] open coding saved:", open_xlsx)

    # =====================================================
    # Step 2) Filtering
    # =====================================================
    all_codes = [
        str(c).strip()
        for c in open_df["open_code"].tolist()
        if isinstance(c, str) and str(c).strip()
    ]
    unique_codes = list(dict.fromkeys(all_codes))  # 保序去重
    print(f"[Module2] open_code total={len(all_codes)}, unique={len(unique_codes)}")

    id2retain, id2reason, id2code = filter_unique_codes_with_batching(
        unique_codes=unique_codes,
        base_url=base_url,
        api_key=api_key,
        model=model_filter,
        system_prompt=SYSTEM_PROMPT_FILTER,
        batch_size=60,
    )

    row_df, unique_df, retain_unique_df, exclude_unique_df = build_filter_outputs_from_open_df(
        open_df=open_df,
        id2retain=id2retain,
        id2reason=id2reason,
        id2code=id2code,
    )

    row_path = os.path.join(out_filter_dir, "open_code_filter_row_level.xlsx")
    unique_all_path = os.path.join(out_filter_dir, "open_code_unique_with_filter.xlsx")
    retain_path = os.path.join(out_filter_dir, "open_code_retain_unique.xlsx")
    exclude_path = os.path.join(out_filter_dir, "open_code_exclude_unique.xlsx")

    row_df.to_excel(row_path, index=False)
    unique_df.to_excel(unique_all_path, index=False)
    retain_unique_df.to_excel(retain_path, index=False)
    exclude_unique_df.to_excel(exclude_path, index=False)

    print("✅ [Module2] filtering saved:")
    print(" -", row_path)
    print(" -", unique_all_path)
    print(" -", retain_path)
    print(" -", exclude_path)

    # =====================================================
    # Step 3) Axial Coding
    # =====================================================
    retain_unique_df = retain_unique_df[retain_unique_df["retain"] == True].copy()
    retain_unique_df["code_id"] = retain_unique_df["code_id"].astype(int)
    retain_unique_df["open_code"] = retain_unique_df["open_code"].astype(str).str.strip()

    id2axial = deepseek_axial_coding(
        retain_unique_df=retain_unique_df[["code_id", "open_code"]],
        base_url=base_url,
        api_key=api_key,
        model=model_axial,
        system_prompt=SYSTEM_PROMPT_AXIAL,
    )

    retain_with_axial = attach_axial_to_retain_unique(retain_unique_df, id2axial)
    axial_unique_path = os.path.join(out_axial_dir, "open_code_retain_unique_axial.xlsx")
    retain_with_axial.to_excel(axial_unique_path, index=False)

    axial_summary = make_axial_summary(retain_with_axial)
    axial_summary_path = os.path.join(out_axial_dir, "axial_coding_summary.xlsx")
    axial_summary.to_excel(axial_summary_path, index=False)

    row_with_axial = attach_axial_to_row_level(row_df, retain_with_axial)
    axial_row_path = os.path.join(out_axial_dir, "axial_coding_row_level.xlsx")
    row_with_axial.to_excel(axial_row_path, index=False)

    print("✅ [Module3] axial coding saved:")
    print(" -", axial_unique_path)
    print(" -", axial_summary_path)
    print(" -", axial_row_path)

    # =====================================================
    # Step 4) Selective Coding
    # =====================================================
    selective_result = deepseek_selective_coding(
        axial_summary_df=axial_summary,
        base_url=base_url,
        api_key=api_key,
        model=model_reasoner,
        system_prompt=SYSTEM_PROMPT_SELECTIVE,
    )

    selective_raw = selective_result.pop("_raw_text", "")
    selective_raw_path = os.path.join(out_selective_dir, "selective_coding_raw.txt")
    with open(selective_raw_path, "w", encoding="utf-8") as f:
        f.write(selective_raw)

    axial_codes = axial_summary["axial_code"].astype(str).str.strip().tolist()
    miss, extra, dup = validate_coverage(selective_result, axial_codes)
    if miss or extra or dup:
        selective_result["_coverage_warning"] = {
            "missing_axial_codes": miss,
            "extra_axial_codes_in_output": extra,
            "duplicated_axial_codes": dup,
        }

    selective_json_path = os.path.join(out_selective_dir, "selective_coding_agg_only.json")
    with open(selective_json_path, "w", encoding="utf-8") as f:
        json.dump(selective_result, f, ensure_ascii=False, indent=2)

    selective_xlsx_path = os.path.join(out_selective_dir, "selective_coding_agg_only.xlsx")
    pd.DataFrame(selective_result.get("aggregate_concepts", []) or []).to_excel(selective_xlsx_path, index=False)

    print("✅ [Module4] selective coding saved:")
    print(" -", selective_raw_path)
    print(" -", selective_json_path)
    print(" -", selective_xlsx_path)

    # =====================================================
    # Step 5) Storyline
    # =====================================================
    storyline_result, storyline_raw = generate_storyline(
        selective_json=selective_result,
        axial_summary_df=axial_summary,
        base_url=base_url,
        api_key=api_key,
        model=model_reasoner,
        system_prompt_storyline=SYSTEM_PROMPT_STORYLINE,
        max_open_examples_per_axial=6,
        timeout=420,
    )

    story_raw_path = os.path.join(out_story_dir, "storyline_raw.txt")
    with open(story_raw_path, "w", encoding="utf-8") as f:
        f.write(storyline_raw)

    validate_storyline_result(storyline_result)

    story_txt_path = os.path.join(out_story_dir, "storyline.txt")
    with open(story_txt_path, "w", encoding="utf-8") as f:
        f.write(str(storyline_result["storyline"]).strip())

    story_json_path = os.path.join(out_story_dir, "storyline.json")
    with open(story_json_path, "w", encoding="utf-8") as f:
        json.dump(storyline_result, f, ensure_ascii=False, indent=2)

    print("✅ [Module5] storyline saved:")
    print(" -", story_raw_path)
    print(" -", story_txt_path)
    print(" -", story_json_path)

    print("\n🎯 All modules finished successfully.\n")


if __name__ == "__main__":
    main()
