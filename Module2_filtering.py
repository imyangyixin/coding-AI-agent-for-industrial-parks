import re
import json
import time
from typing import Any, Dict, List, Tuple, Optional

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


def _call_filter_batch(
    *,
    batch_items: List[Dict[str, Any]],
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    timeout: int = 180,
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload_codes = {"open_codes": batch_items}
    codes_json_str = json.dumps(payload_codes, ensure_ascii=False, separators=(",", ":"))

    user_content = (
        "下面是一组 open_code（JSON）：\n"
        f"{codes_json_str}\n\n"
        "请按照系统提示，只做相关性筛选，并严格按 JSON 格式输出。"
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
    raw_text = resp.json()["choices"][0]["message"]["content"]
    return safe_parse_json(raw_text) or {}


def _normalize_filtering_result(data: Dict[str, Any], batch_size: int) -> List[Dict[str, Any]]:
    if not data or "filtering" not in data or not isinstance(data["filtering"], list):
        return []

    seen = set()
    out = []
    for item in data["filtering"]:
        try:
            bid = int(item.get("id"))
        except Exception:
            continue
        if bid < 1 or bid > batch_size or bid in seen:
            continue
        seen.add(bid)

        retain = bool(item.get("retain"))
        reason = str(item.get("exclude_reason", "")).strip()
        if retain:
            reason = ""
        out.append({"id": bid, "retain": retain, "exclude_reason": reason})

    for bid in range(1, batch_size + 1):
        if bid not in seen:
            out.append({"id": bid, "retain": False, "exclude_reason": "模型未返回该条结果，需人工检查"})

    out.sort(key=lambda x: x["id"])
    return out


def filter_unique_codes_with_batching(
    *,
    unique_codes: List[str],
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    batch_size: int = 60,
    between_batch_sleep: float = 1.0,
    max_retries_each_batch: int = 2,
    retry_sleep: float = 2.0,
) -> Tuple[Dict[int, bool], Dict[int, str], Dict[int, str]]:
    id2code = {i + 1: code for i, code in enumerate(unique_codes)}
    id2retain: Dict[int, bool] = {}
    id2reason: Dict[int, str] = {}

    def process_range(start_idx: int, end_idx: int):
        sub = unique_codes[start_idx:end_idx]
        global_ids = list(range(start_idx + 1, end_idx + 1))
        n = len(sub)
        if n == 0:
            return

        batch_items = [{"id": i + 1, "text": sub[i]} for i in range(n)]

        last_err = None
        for attempt in range(1, max_retries_each_batch + 1):
            try:
                data = _call_filter_batch(
                    batch_items=batch_items,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    system_prompt=system_prompt,
                )
                filtering = _normalize_filtering_result(data, n)
                if not filtering:
                    raise ValueError("解析失败/无 filtering")

                for item in filtering:
                    bid = int(item["id"])
                    gid = global_ids[bid - 1]
                    id2retain[gid] = bool(item["retain"])
                    id2reason[gid] = str(item["exclude_reason"]).strip()
                return
            except Exception as e:
                last_err = e
                if attempt < max_retries_each_batch:
                    time.sleep(retry_sleep)

        if n == 1:
            gid = global_ids[0]
            id2retain[gid] = False
            id2reason[gid] = f"API失败/解析失败（单条）：{repr(last_err)}"
            return

        mid = start_idx + n // 2
        process_range(start_idx, mid)
        process_range(mid, end_idx)

    total = len(unique_codes)
    for s in range(0, total, batch_size):
        e = min(s + batch_size, total)
        process_range(s, e)
        time.sleep(between_batch_sleep)

    return id2retain, id2reason, id2code


def build_filter_outputs_from_open_df(
    *,
    open_df: pd.DataFrame,
    id2retain: Dict[int, bool],
    id2reason: Dict[int, str],
    id2code: Dict[int, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_df = pd.DataFrame(
        [
            {
                "code_id": cid,
                "open_code": txt,
                "retain": bool(id2retain.get(cid, False)),
                "exclude_reason": str(id2reason.get(cid, "未返回该条结果")).strip(),
            }
            for cid, txt in id2code.items()
        ]
    )

    retain_unique_df = unique_df[unique_df["retain"] == True].copy()
    exclude_unique_df = unique_df[unique_df["retain"] == False].copy()

    code2id = {v: k for k, v in id2code.items()}

    def map_code_id(x):
        if not isinstance(x, str):
            return ""
        return code2id.get(x.strip(), "")

    def map_retain(x):
        if not isinstance(x, str):
            return False
        cid = code2id.get(x.strip(), None)
        return bool(id2retain.get(cid, False)) if cid else False

    def map_reason(x):
        if not isinstance(x, str):
            return ""
        cid = code2id.get(x.strip(), None)
        return id2reason.get(cid, "open_code 未在 unique 集合中") if cid else "open_code 未在 unique 集合中"

    row_df = open_df.copy()
    row_df["code_id"] = row_df["open_code"].apply(map_code_id)
    row_df["retain"] = row_df["open_code"].apply(map_retain)
    row_df["exclude_reason"] = row_df["open_code"].apply(map_reason)

    return row_df, unique_df, retain_unique_df, exclude_unique_df
