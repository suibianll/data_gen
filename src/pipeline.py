"""End-to-end pipeline: extract → generate records → build Q&A."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from tqdm import tqdm

from .extractor import MedicalInfoExtractor
from .llm_client import LLMClient
from .qa_builder import QABuilder, record_to_text
from .record_generator import MedicalRecordGenerator

logger = logging.getLogger(__name__)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", path)


class Pipeline:
    """Orchestrate extraction, record generation, and QA building.

    Parameters
    ----------
    config:
        Loaded configuration dict (from ``config.yaml``).
    """

    def __init__(self, config: dict) -> None:
        llm_cfg = config.get("llm", {})
        self.llm = LLMClient(
            api_key=llm_cfg.get("api_key", ""),
            base_url=llm_cfg.get("base_url", "https://api.openai.com/v1"),
            model=llm_cfg.get("model", "gpt-4o"),
            temperature=float(llm_cfg.get("temperature", 0.7)),
        )
        paths = config.get("paths", {})
        self.input_dir = Path(paths.get("input_dir", "data/input"))
        self.output_dir = Path(paths.get("output_dir", "data/output"))
        gen = config.get("generation", {})
        self.num_records_per_disease = int(gen.get("num_records_per_disease", 3))
        self.num_qa_pairs = int(gen.get("num_qa_pairs", 2))

        self.extractor = MedicalInfoExtractor(self.llm)
        self.record_gen = MedicalRecordGenerator(self.llm)
        self.qa_builder = QABuilder(self.llm)

    # ------------------------------------------------------------------
    # Step 1 – Extract
    # ------------------------------------------------------------------

    def extract(self, input_file: str | Path) -> List[dict]:
        """Extract disease entries from *input_file* and save them.

        Returns
        -------
        list of dict
        """
        logger.info("Extracting medical info from %s …", input_file)
        diseases = self.extractor.extract_from_file(input_file)
        out_path = self.output_dir / "extracted_diseases.json"
        _write_json(out_path, diseases)
        logger.info("Extracted %d disease entries.", len(diseases))
        return diseases

    # ------------------------------------------------------------------
    # Step 2 – Generate records
    # ------------------------------------------------------------------

    def generate_records(self, diseases: List[dict]) -> List[dict]:
        """Generate synthetic medical records for every disease entry.

        Returns
        -------
        list of dict
            Each entry has ``disease_info`` and ``records`` (list of record dicts).
        """
        all_results: List[dict] = []
        for disease_info in tqdm(diseases, desc="Generating records"):
            logger.info(
                "Generating %d record(s) for: %s",
                self.num_records_per_disease,
                disease_info["disease"],
            )
            records = self.record_gen.generate_batch(
                disease_info, n=self.num_records_per_disease
            )
            all_results.append({"disease_info": disease_info, "records": records})

        out_path = self.output_dir / "generated_records.json"
        _write_json(out_path, all_results)
        return all_results

    # ------------------------------------------------------------------
    # Step 3 – Build QA pairs
    # ------------------------------------------------------------------

    def build_qa(self, record_results: List[dict]) -> List[dict]:
        """Build Q&A pairs for every generated medical record.

        Returns
        -------
        list of dict
            Each entry has ``question``, ``record``, ``answer``, ``golden_chunks``.
        """
        all_qa: List[dict] = []
        for entry in tqdm(record_results, desc="Building QA pairs"):
            disease_name = entry["disease_info"]["disease"]
            source = entry["disease_info"].get("source", disease_name)
            for record in entry["records"]:
                logger.info("Building QA for disease: %s", disease_name)
                try:
                    qa_pairs = self.qa_builder.build(record, n=self.num_qa_pairs)
                except (ValueError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "QA generation failed for %s: %s", disease_name, exc
                    )
                    qa_pairs = []
                record_text = record_to_text(record)
                for qa in qa_pairs:
                    golden_chunks = qa.get("golden_chunks", [])
                    if isinstance(golden_chunks, list):
                        for chunk in golden_chunks:
                            if isinstance(chunk, dict) and not chunk.get("source"):
                                chunk["source"] = source
                    all_qa.append(
                        {
                            "question": qa.get("question", ""),
                            "record": record_text,
                            "answer": qa.get("answer", ""),
                            "golden_chunks": golden_chunks,
                        }
                    )

        out_path = self.output_dir / "qa_dataset.json"
        _write_json(out_path, all_qa)
        return all_qa

    # ------------------------------------------------------------------
    # Run all steps
    # ------------------------------------------------------------------

    def run(self, input_file: str | Path) -> List[dict]:
        """Execute the full pipeline: extract → generate → QA.

        Parameters
        ----------
        input_file:
            Path to the input medical dataset file.

        Returns
        -------
        list of dict
            The final QA dataset.
        """
        diseases = self.extract(input_file)
        record_results = self.generate_records(diseases)
        qa_dataset = self.build_qa(record_results)
        logger.info(
            "Pipeline complete. %d QA samples written to %s",
            len(qa_dataset),
            self.output_dir / "qa_dataset.json",
        )
        return qa_dataset
