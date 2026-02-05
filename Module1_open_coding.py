import re
import json
import time
from typing import List, Dict, Optional

import requests
import pandas as pd


def parse_qa_blocks(text: str) -> List[Dict[str, str]]:
    blocks = []
    current_q = None
    current_a_lines: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith(("Q:", "Q：")):
            if current_q is not None and current_a_lines:
                blocks.append({"question": current_q.strip(), "answer": "\n".join(current_a_lines).strip()})
                current_a_lines = []
            current_q = re.sub(r"^Q[:：]\s*", "", line)

        elif line.startswith(("A:", "A：")):
            a_body = re.sub(r"^A[:：]\s*", "", line)
            if a_body:
                current_a_lines.append(a_body)

        else:
            if current_a_lines:
                current_a_lines.append(line)
            elif current_q is not None:
                current_q += " " + line

    if current_q is not None and current_a_lines:
        blocks.append({"question": current_q.strip(), "answer": "\n".join(current_a_lines).strip()})

    return blocks


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


def open_code_answer(
    *,
    question: str,
    answer: str,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    max_retries: int = 3,
    retry_sleep: float = 2.0,
    timeout: int = 120,
) -> str:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    user_content = (
        "下面是一段访谈问答片段。\n"
        f"【问题】：{question}\n"
        f"【回答】：{answer}\n\n"
        "请仅基于【回答】的内容进行开放性编码，问题只用来帮助你理解语境，不要对问题本身编码。"
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
            raw_text = resp.json()["choices"][0]["message"]["content"].strip()

            obj = safe_parse_json(raw_text)
            if isinstance(obj, dict) and "open_code" in obj:
                return str(obj["open_code"]).strip()

            m = re.search(r'"open_code"\s*:\s*"([^"]+)"', raw_text)
            if m:
                return m.group(1).strip()

            return raw_text

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(retry_sleep)

    return f"[API 调用多次失败: {repr(last_err)}]"


def run_open_coding_from_text(
    *,
    text: str,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    sleep_time: float = 1.0,
) -> pd.DataFrame:
    qa_blocks = parse_qa_blocks(text)
    rows = []

    for idx, block in enumerate(qa_blocks, start=1):
        code = open_code_answer(
            question=block["question"],
            answer=block["answer"],
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
        )
        rows.append({"id": idx, "question": block["question"], "answer": block["answer"], "open_code": code})
        time.sleep(sleep_time)

    return pd.DataFrame(rows)
