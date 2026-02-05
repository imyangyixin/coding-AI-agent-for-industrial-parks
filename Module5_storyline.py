import re
import json
from typing import Dict, Any, List, Optional, Tuple

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
        start = s.find("{")
        end = s.rfind("}") + 1
        try:
            return json.loads(s[start:end])
        except Exception:
            return None
    return None


def deepseek_chat(
    *,
    system_prompt: str,
    user_content: str,
    base_url: str,
    api_key: str,
    model: str,
    timeout: int = 420,
) -> str:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
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
    return resp.json()["choices"][0]["message"]["content"]


def pick_examples_from_member_text(
    member_text: str,
    max_items: int = 6,
    max_chars_each: int = 24,
) -> List[str]:
    """
    member_text: 形如 "xxx; yyy; zzz" 或一长段文本
    策略：按 ; / 、 / 换行 / | / , 分割，去空去重，截断长度
    """
    if not isinstance(member_text, str):
        return []
    s = member_text.replace("\n", " ").strip()
    if not s:
        return []
    parts = re.split(r"[;；、]\s*|\|\s*|,\s*", s)
    parts = [p.strip() for p in parts if p and p.strip()]

    seen: List[str] = []
    for p in parts:
        if p not in seen:
            seen.append(p)

    out: List[str] = []
    for p in seen[: max_items * 3]:
        p2 = p[:max_chars_each]
        if p2 and p2 not in out:
            out.append(p2)
        if len(out) >= max_items:
            break
    return out


def load_selective_aggregate_from_dict(selective_json: Dict[str, Any]) -> Dict[str, Any]:
    if "aggregate_concepts" not in selective_json:
        raise ValueError("selective JSON 缺少 aggregate_concepts")
    return selective_json


def load_axial_summary_df(axial_summary_df: pd.DataFrame) -> pd.DataFrame:
    df = axial_summary_df.copy()
    if "axial_code" not in df.columns:
        raise ValueError("axial summary 缺少 axial_code 列")

    if "member_open_codes" not in df.columns:
        possible = [c for c in df.columns if "member" in c and "code" in c]
        if possible:
            df = df.rename(columns={possible[0]: "member_open_codes"})
        else:
            df["member_open_codes"] = ""

    df["axial_code"] = df["axial_code"].astype(str).str.strip()
    df = df[df["axial_code"] != ""].copy()
    return df


def build_one_shot_payload(
    *,
    selective: Dict[str, Any],
    axial_df: pd.DataFrame,
    max_open_examples_per_axial: int = 6,
) -> str:
    # 1) aggregate concepts（保留必要字段）
    agg = []
    for c in selective.get("aggregate_concepts", []) or []:
        agg.append(
            {
                "concept": str(c.get("concept", "")).strip(),
                "definition": str(c.get("definition", "")).strip(),
                "covered_axial_codes": [
                    str(x).strip() for x in (c.get("covered_axial_codes", []) or []) if str(x).strip()
                ],
            }
        )

    # 2) axial themes + open examples（压缩）
    axial_items = []
    for _, row in axial_df.iterrows():
        ax = str(row.get("axial_code", "")).strip()
        mem = str(row.get("member_open_codes", "") or "")
        examples = pick_examples_from_member_text(mem, max_items=max_open_examples_per_axial, max_chars_each=28)
        axial_items.append({"axial_code": ax, "open_code_examples": examples})

    payload = {"aggregate_concepts": agg, "axial_themes": axial_items}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def generate_storyline(
    *,
    selective_json: Dict[str, Any],
    axial_summary_df: pd.DataFrame,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt_storyline: str,
    max_open_examples_per_axial: int = 6,
    timeout: int = 420,
) -> Tuple[Dict[str, Any], str]:
    """
    返回：
      - result: {"storyline": "...", "anchors": [...] , ...}
      - raw_text: 模型原始输出（方便落盘排查）
    """
    selective = load_selective_aggregate_from_dict(selective_json)
    axial_df = load_axial_summary_df(axial_summary_df)

    payload_json = build_one_shot_payload(
        selective=selective,
        axial_df=axial_df,
        max_open_examples_per_axial=max_open_examples_per_axial,
    )

    user_content = (
        "以下是三层编码结果（选择性编码 aggregate_concepts + 主轴编码 axial_themes（含少量 open_code 例证））：\n"
        f"{payload_json}\n\n"
        "请严格按系统提示输出 JSON（storyline + anchors）。"
    )

    raw = deepseek_chat(
        system_prompt=system_prompt_storyline,
        user_content=user_content,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
    )

    result = safe_parse_json(raw) or {}
    return result, raw


def validate_storyline_result(result: Dict[str, Any]) -> None:
    if "storyline" not in result or not str(result.get("storyline", "")).strip():
        raise RuntimeError("返回缺少 storyline 或 storyline 为空")
    if "anchors" not in result or not isinstance(result["anchors"], list) or len(result["anchors"]) == 0:
        raise RuntimeError("返回缺少 anchors 或 anchors 为空")
