import re
import json
import time
from typing import Dict, Optional

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


def deepseek_axial_coding(
    *,
    retain_unique_df: pd.DataFrame,  # 必含 code_id, open_code
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    max_retries: int = 3,
    retry_sleep: float = 3.0,
    sleep_time: float = 1.0,
    timeout: int = 240,
) -> Dict[int, str]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    id2code = {int(r["code_id"]): str(r["open_code"]).strip() for _, r in retain_unique_df.iterrows()}
    payload_codes = {"open_codes": [{"id": cid, "text": txt} for cid, txt in id2code.items()]}
    codes_json_str = json.dumps(payload_codes, ensure_ascii=False, separators=(",", ":"))

    user_content = (
        "以下是保留的 open_code（JSON）：\n"
        f"{codes_json_str}\n\n"
        "请按系统提示进行主轴编码，并严格输出 JSON。"
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

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            raw_text = resp.json()["choices"][0]["message"]["content"]
            data = safe_parse_json(raw_text)

            if data and "axial_coding" in data and isinstance(data["axial_coding"], list):
                id2axial: Dict[int, str] = {}
                for g in data["axial_coding"]:
                    axial = str(g.get("axial_code", "")).strip()
                    for cid in g.get("member_ids", []) or []:
                        try:
                            cid = int(cid)
                        except Exception:
                            continue
                        id2axial[cid] = axial
                time.sleep(sleep_time)
                return id2axial
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(retry_sleep)

    return {}


def attach_axial_to_retain_unique(retain_unique_df: pd.DataFrame, id2axial: Dict[int, str]) -> pd.DataFrame:
    out = retain_unique_df.copy()
    out["axial_code"] = out["code_id"].apply(lambda cid: id2axial.get(int(cid), ""))
    return out


def make_axial_summary(retain_unique_with_axial: pd.DataFrame) -> pd.DataFrame:
    df = retain_unique_with_axial.copy()
    df["axial_code"] = df["axial_code"].astype(str).str.strip()
    df = df[df["axial_code"] != ""].copy()

    summary = (
        df.groupby("axial_code")["open_code"]
        .apply(lambda x: "; ".join(sorted(set(x))))
        .reset_index()
        .rename(columns={"open_code": "member_open_codes"})
    )
    summary["n_members"] = summary["member_open_codes"].apply(
        lambda s: 0 if not isinstance(s, str) or not s.strip() else len(s.split("; "))
    )
    return summary


def attach_axial_to_row_level(row_df: pd.DataFrame, retain_unique_with_axial: pd.DataFrame) -> pd.DataFrame:
    # code_id -> axial_code
    mapping = {int(r["code_id"]): str(r["axial_code"]).strip() for _, r in retain_unique_with_axial.iterrows()}

    out = row_df.copy()

    def map_axial(x):
        try:
            cid = int(x)
        except Exception:
            return ""
        return mapping.get(cid, "")

    out["axial_code"] = out["code_id"].apply(map_axial)
    return out
