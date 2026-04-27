"""Extract diseases, symptoms, and treatment methods from a medical dataset file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .llm_client import LLMClient
from .utils import clean_json

_SYSTEM_PROMPT = """\
你是一名经验丰富的医学信息抽取专家。请从给定的医学文本中提取以下结构化信息：
1. 病症名称（disease）
2. 该病症的主要临床表现与症状（symptoms）
3. 该病症的常用治疗方法与用药方案（treatments）

请以 JSON 数组的形式输出，每个元素包含以下字段：
{
  "disease": "病症名称",
  "symptoms": ["症状1", "症状2", ...],
  "treatments": ["治疗方法1", "治疗方法2", ...]
}

只输出 JSON，不要输出任何其他文字。
"""



class MedicalInfoExtractor:
    """Use an LLM to extract structured medical information from a text file.

    Parameters
    ----------
    llm_client:
        Configured :class:`LLMClient` instance.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def extract_from_text(self, text: str) -> List[dict]:
        """Extract disease entries from an arbitrary medical text string.

        Parameters
        ----------
        text:
            Raw medical dataset content.

        Returns
        -------
        list of dict
            Each dict has keys ``disease``, ``symptoms``, ``treatments``.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"请从以下医学文本中提取病症信息：\n\n{text}"},
        ]
        raw = self.llm.chat(messages, temperature=0.2)
        cleaned = clean_json(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned invalid JSON during extraction.\nRaw output:\n{raw}"
            ) from exc
        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array from LLM, got: {type(data).__name__}"
            )
        return data

    def extract_from_file(self, file_path: str | Path) -> List[dict]:
        """Read a file and extract medical information from its contents.

        Parameters
        ----------
        file_path:
            Path to the input medical dataset file.

        Returns
        -------
        list of dict
        """
        text = Path(file_path).read_text(encoding="utf-8")
        return self.extract_from_text(text)
