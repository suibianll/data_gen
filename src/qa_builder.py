"""Build Q&A pairs from a generated medical record."""

from __future__ import annotations

import json
from typing import List

from .llm_client import LLMClient
from .utils import clean_json

_SYSTEM_PROMPT = """\
你是一名医学教育专家，擅长根据病历设计高质量的病历审查问题及其参考答案。

你需要从两个维度提问：
1. 病历书写合规性（Query1）：从病历书写规范的角度，分析该病历是否有前后矛盾、缺失项目、逻辑不一致等问题。
2. 诊疗逻辑合理性（Query2）：从医学逻辑的角度，分析该患者的诊断是否与症状相符、用药是否有禁忌、诊疗思路是否合理。

对于每个问题，请提供：
- 问题（question）
- 参考答案（answer）
- 参考段落（reference）：直接引用病历中支持该答案的相关原文片段（若有多处，请列出多处）

请以 JSON 数组的形式输出，每个元素结构如下：
{
  "query_type": "Query1" 或 "Query2",
  "question": "...",
  "answer": "...",
  "references": ["病历原文片段1", "病历原文片段2", ...]
}

只输出 JSON，不要输出任何其他文字。
"""

_USER_TEMPLATE = """\
请根据以下病历内容，生成 {n} 组问答对（Query1 和 Query2 各至少一个）：

{record_text}
"""


def _record_to_text(record: dict) -> str:
    """Convert a structured medical record dict to a readable Chinese text."""
    lines: List[str] = []

    basic = record.get("基本信息", {})
    if basic:
        lines.append("【基本信息】")
        for k, v in basic.items():
            lines.append(f"  {k}：{v}")

    for field in ["主诉", "现病史", "其他疾病", "在服药物", "既往史", "个人史", "婚育史", "家族史"]:
        val = record.get(field)
        if val:
            lines.append(f"【{field}】{val}")

    exam = record.get("体格检查", {})
    if exam:
        lines.append("【体格检查】")
        for k, v in exam.items():
            lines.append(f"  {k}：{v}")

    aux = record.get("辅助检查")
    if aux:
        lines.append(f"【辅助检查】{aux}")

    diags = record.get("初步诊断", [])
    if diags:
        lines.append("【初步诊断】" + "；".join(diags))

    return "\n".join(lines)


class QABuilder:
    """Generate Q&A pairs with reference paragraphs from a medical record.

    Parameters
    ----------
    llm_client:
        Configured :class:`LLMClient` instance.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def build(self, record: dict, n: int = 2) -> List[dict]:
        """Generate *n* Q&A pairs from a medical record.

        Parameters
        ----------
        record:
            Medical record as a dict (output from
            :class:`~src.record_generator.MedicalRecordGenerator`).
        n:
            Minimum number of Q&A pairs to request.  The LLM may return more.

        Returns
        -------
        list of dict
            Each dict has keys ``query_type``, ``question``, ``answer``,
            ``references``.
        """
        record_text = _record_to_text(record)
        user_msg = _USER_TEMPLATE.format(n=n, record_text=record_text)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        raw = self.llm.chat(messages, temperature=0.5)
        cleaned = clean_json(raw)
        try:
            qa_pairs = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned invalid JSON for QA generation.\nRaw output:\n{raw}"
            ) from exc
        if not isinstance(qa_pairs, list):
            raise ValueError(
                f"Expected a JSON array from LLM, got: {type(qa_pairs).__name__}"
            )
        return qa_pairs
