"""Generate realistic Chinese medical admission records (病历) using an LLM."""

from __future__ import annotations

import json
from typing import List

from .llm_client import LLMClient
from .utils import clean_json

_SYSTEM_PROMPT = """\
你是一名资深医疗文书撰写专家，擅长撰写规范的中文医院入院记录。

在生成病历时，请严格遵循以下规范：
1. 病历格式完整，包含：基本信息、主诉、现病史、既往史、个人史、婚育史、家族史、体格检查、辅助检查、初步诊断、鉴别诊断、诊疗计划、病情评估。
2. 内容前后一致，无逻辑矛盾（如年龄、性别、病程、检查结果互相吻合）。
3. 主诉与现病史紧密相关；辅助检查结果支持初步诊断；用药与诊断相符，无明显禁忌。
4. 数值指标符合临床实际（如血压、体温、心率等正常或异常范围合理）。
5. 体现临床推理链：症状→检查→诊断→治疗计划，需给出关键风险点与监测重点。
6. 语言专业、简洁，符合中国医院病历书写习惯。

请以如下 JSON 格式输出一份病历，字段值均为中文字符串：
{
  "基本信息": {
    "性别": "...",
    "年龄": "...",
    "民族": "...",
    "婚姻": "...",
    "职业": "...",
    "供史者": "...",
    "入院时间": "...",
    "记录时间": "..."
  },
  "主诉": "...",
  "现病史": "...",
  "其他疾病": "...",
  "在服药物": "...",
  "既往史": "...",
  "个人史": "...",
  "婚育史": "...",
  "家族史": "...",
  "体格检查": {
    "体温": "...",
    "脉搏": "...",
    "呼吸": "...",
    "血压": "...",
    "一般情况": "..."
  },
  "辅助检查": "...",
  "初步诊断": ["主要诊断", "次要诊断（如有）"],
  "鉴别诊断": ["诊断A（及排除理由）", "诊断B（及排除理由）"],
  "诊疗计划": "...",
  "病情评估": "..."
}

只输出 JSON，不要输出任何其他文字。
"""

_USER_TEMPLATE = """\
请根据以下病症信息生成一份真实、规范的中文入院病历：

病症名称：{disease}
主要症状与临床表现：
{symptoms}
常用治疗方法：
{treatments}

要求：
- 病历中的患者为虚构人物，性别、年龄请随机生成且符合该病症的典型人群特征。
- 辅助检查结果应与症状及诊断一致。
- 初步诊断应准确反映上述病症，并可酌情添加并发症或合并症。
- 优先加入复杂但合理的临床情境（如合并基础病、用药相互作用风险、并发症监测点）。
"""


def _render_list(items: list) -> str:
    return "\n".join(f"- {item}" for item in items)


class MedicalRecordGenerator:
    """Generate synthetic Chinese medical admission records.

    Parameters
    ----------
    llm_client:
        Configured :class:`LLMClient` instance.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def generate(self, disease_info: dict) -> dict:
        """Generate a single medical record from a disease-info dict.

        Parameters
        ----------
        disease_info:
            Dict with keys ``disease``, ``symptoms`` (list), ``treatments`` (list).

        Returns
        -------
        dict
            Parsed medical record as a Python dict matching the JSON schema above.
        """
        user_msg = _USER_TEMPLATE.format(
            disease=disease_info["disease"],
            symptoms=_render_list(disease_info.get("symptoms", [])),
            treatments=_render_list(disease_info.get("treatments", [])),
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        raw = self.llm.chat(messages)
        cleaned = clean_json(raw)
        try:
            record = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned invalid JSON for record generation.\nRaw output:\n{raw}"
            ) from exc
        return record

    def generate_batch(
        self, disease_info: dict, n: int = 1
    ) -> List[dict]:
        """Generate *n* distinct medical records for the same disease.

        Parameters
        ----------
        disease_info:
            Disease information dict.
        n:
            Number of records to generate.

        Returns
        -------
        list of dict
        """
        return [self.generate(disease_info) for _ in range(n)]
