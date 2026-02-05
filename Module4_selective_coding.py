import re
import json
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import requests


def safe_parse_json(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    s = re.sub(r"^```json\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    try:
        return json.loads(s)
    except Exception:
        pass

    if "{" in s and "}" in s:
        snippet = s[s.find("{"): s.rfind("}") + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None


def build_selective_payload(axial_summary_df: pd.DataFrame, example_char_limit: int = 220) -> str:
    items = []
    for _, row in axial_summary_df.iterrows():
        axial_code = str(row.get("axial_code", "")).strip()
        if not axial_code:
            continue
        ex = str(row.get("member_open_codes", "") or "").strip().replace("\n", " ")
        ex = ex[:example_char_limit] if example_char_limit > 0 else ""
        items.append({"axial_code": axial_code, "member_open_codes_excerpt": ex})

    return json.dumps({"axial_items": items}, ensure_ascii=False, separators=(",", ":"))


def deepseek_selective_coding(
    *,
    axial_summary_df: pd.DataFrame,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    timeout: int = 300,
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload_json = build_selective_payload(axial_summary_df, example_char_limit=220)

    user_content = (
        "以下是全部主轴编码（axial_code）及少量例证（可能截断）：\n"
        f"{payload_json}\n\n"
        "请基于这些 axial_code 提炼聚合概念。必须覆盖全部 axial_code，每个 axial_code 必须且只能归入一个聚合概念；输出只允许 JSON。"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "stream": False,
    }

    resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]

    result = safe_parse_json(raw) or {}
    result["_raw_text"] = raw  # 方便 run.py 落盘 raw
    return result


def validate_coverage(result: Dict[str, Any], axial_codes: List[str]) -> Tuple[List[str], List[str], List[str]]:
    axial_set = set(axial_codes)
    seen = []
    for c in result.get("aggregate_concepts", []) or []:
        for a in c.get("covered_axial_codes", []) or []:
            seen.append(str(a).strip())

    seen_set = set(seen)
    missing = [a for a in axial_codes if a not in seen_set]
    extra = [a for a in seen_set if a not in axial_set]

    from collections import Counter
    cnt = Counter(seen)
    dup = [k for k, v in cnt.items() if v > 1 and k]
    return missing, extra, dup