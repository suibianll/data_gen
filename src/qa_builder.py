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
3. 高难度临床推理（Query3）：结合时间线、并发症风险、鉴别诊断、用药安全、指南一致性进行综合分析。

对于每个问题，请提供：
- 问题（question）
- 参考答案（answer）
- 证据片段（golden_chunks）：直接引用病历中支持该答案的相关原文片段，并给出来源字段名

请以 JSON 数组的形式输出，每个元素结构如下：
{
  "question": "...",
  "answer": "...",
  "golden_chunks": [
    {"content": "病历原文片段1", "source": "现病史"},
    {"content": "病历原文片段2", "source": "辅助检查"}
  ]
}

只输出 JSON，不要输出任何其他文字。
"""

_USER_TEMPLATE = """\
请根据以下病历内容，生成 {n} 组高质量问答对，要求：
- 至少包含 1 个 Query1、1 个 Query2、1 个 Query3 风格的问题；
- 问题要有一定难度，优先考察矛盾识别、诊疗链条完整性、禁忌风险判断；
- 每个答案都必须引用 golden_chunks，且 source 要尽量精确到病历字段。

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
            Each dict has keys ``question``, ``answer``, ``golden_chunks``.
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
        normalized: List[dict] = []
        for qa in qa_pairs:
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question", "")).strip()
            answer = str(qa.get("answer", "")).strip()
            chunks = qa.get("golden_chunks")
            if not isinstance(chunks, list):
                refs = qa.get("references", [])
                chunks = (
                    [{"content": str(ref), "source": "病历原文"} for ref in refs if str(ref).strip()]
                    if isinstance(refs, list)
                    else []
                )
            normalized_chunks = []
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                content = str(chunk.get("content", "")).strip()
                source = str(chunk.get("source", "病历原文")).strip() or "病历原文"
                if content:
                    normalized_chunks.append({"content": content, "source": source})
            if question and answer:
                normalized.append(
                    {
                        "question": question,
                        "answer": answer,
                        "golden_chunks": normalized_chunks,
                    }
                )
        return normalized


def record_to_text(record: dict) -> str:
    """Public helper for rendering record dict to readable text."""
    return _record_to_text(record)
