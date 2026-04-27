# data_gen – Medical Record & QA Data Generation Pipeline

一个面向医疗场景的数据生成工程，基于大语言模型（LLM）从医学数据集中自动生成规范的中文入院病历，并构建配套的问答数据集。

---

## 功能概述

| 阶段 | 说明 |
|------|------|
| **Step 1 – 信息抽取** | 读取医学数据集文件，利用 LLM 提取病症名称、临床症状、治疗方法 |
| **Step 2 – 病历生成** | 针对每种病症，生成若干份规范的中文入院记录（虚构患者） |
| **Step 3 – QA 构建** | 对每份病历生成两类问答对，并附参考段落 |

### 两类问答类型

- **Query1（病历书写合规性）**：分析病历是否符合书写规范，前后有无矛盾、缺失项等
- **Query2（诊疗逻辑合理性）**：分析诊断是否与症状相符、用药有无禁忌、诊疗思路是否合理

---

## 项目结构

```
data_gen/
├── config.yaml                    # 配置文件（API Key、模型、生成参数等）
├── main.py                        # 入口脚本
├── requirements.txt
├── data/
│   ├── input/                     # 放置医学数据集文件（txt / md 等）
│   │   └── sample_medical_dataset.txt
│   └── output/                    # 生成结果
│       ├── extracted_diseases.json
│       ├── generated_records.json
│       └── qa_dataset.json
└── src/
    ├── llm_client.py              # OpenAI-compatible LLM 封装
    ├── extractor.py               # 病症信息抽取
    ├── record_generator.py        # 病历生成
    ├── qa_builder.py              # 问答对构建
    └── pipeline.py                # 流水线编排
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

编辑 `config.yaml`：

```yaml
llm:
  api_key: "your-api-key-here"        # LLM API Key
  base_url: "https://api.openai.com/v1"  # 兼容 OpenAI 格式的任意端点
  model: "gpt-4o"
  temperature: 0.7

paths:
  input_dir: "data/input"
  output_dir: "data/output"

generation:
  num_records_per_disease: 3   # 每种病症生成的病历份数
  num_qa_pairs: 2              # 每份病历生成的问答对数量
```

支持任何兼容 OpenAI API 格式的服务（如本地 Ollama、vLLM、Azure OpenAI 等）。

### 3. 准备输入数据

将医学数据集文件放入 `data/input/` 目录。支持纯文本，也支持如下 JSON 文档（单篇或多篇）：

```json
[
  {
    "file_name": "xxx.pdf",
    "file_type": "PDF",
    "file_path": "/path/to/xxx.pdf",
    "content": "文章解析后的正文内容"
  }
]
```

程序会自动读取 `content` 字段并汇总后抽取病症信息。  
仓库已附带示例文件 `data/input/sample_medical_dataset.txt`（包含消化道出血、COPD、高血压病三个病种）。

### 4. 运行

```bash
# 使用 data/input/ 下第一个文件作为输入
python main.py

# 指定输入文件
python main.py --input data/input/my_dataset.txt

# 指定配置文件
python main.py --config my_config.yaml
```

### 5. 查看输出

```
data/output/
├── extracted_diseases.json   # Step1 抽取结果：病症+症状+治疗
├── generated_records.json    # Step2 生成结果：入院病历
└── qa_dataset.json           # Step3 最终数据集：病历+问答对+参考段落
```

#### `qa_dataset.json` 示例结构

```json
[
  {
    "question": "该病例中初步诊断与辅助检查是否完全一致？可能存在哪些诊疗风险？",
    "record": "【基本信息】...\n【主诉】...\n【现病史】...\n...",
    "answer": "...",
    "golden_chunks": [
      { "content": "入院前自服阿司匹林...", "source": "在服药物" },
      { "content": "粪隐血阳性，Hb 86g/L", "source": "辅助检查" }
    ]
  }
]
```

---

## 模块说明

### `src/llm_client.py`
封装 OpenAI `chat.completions.create`，支持任意兼容端点。

### `src/extractor.py`
将数据集文本发送给 LLM，返回结构化的 `[{disease, symptoms, treatments}]` 列表。

### `src/record_generator.py`
根据病症信息，生成符合中国医院病历书写规范的入院记录 JSON。

### `src/qa_builder.py`
对生成的病历构建 Query1（书写合规性）和 Query2（诊疗逻辑）两类问答对，附参考段落。

### `src/pipeline.py`
串联上述三个步骤，每步结果持久化到 `data/output/`。
