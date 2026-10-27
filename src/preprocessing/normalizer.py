"""
Normalizacao de textos clinicos.
Util para padronizar abreviacoes, unidades e termos medicos.
"""

from typing import Dict
import re


class ClinicalNormalizer:
    """Normaliza abreviacoes e unidades comuns em textos clinicos."""

    def __init__(self):
        self.abbreviations: Dict[str, str] = {
            "aa": "ar ambiente",
            "aas": "acido acetilsalicilico",
            "avc": "acidente vascular cerebral",
            "bpm": "batimentos por minuto",
            "dac": "doenca arterial coronariana",
            "dm": "diabetes mellitus",
            "dm1": "diabetes mellitus tipo 1",
            "dm2": "diabetes mellitus tipo 2",
            "dpo": "dia pos-operatorio",
            "dpoc": "doenca pulmonar obstrutiva cronica",
            "drc": "doenca renal cronica",
            "ecg": "eletrocardiograma",
            "ev": "endovenoso",
            "fc": "frequencia cardiaca",
            "fr": "frequencia respiratoria",
            "hba1c": "hemoglobina glicada",
            "has": "hipertensao arterial sistemica",
            "hb": "hemoglobina",
            "hgt": "hemoglicoteste",
            "ht": "hematocrito",
            "iam": "infarto agudo do miocardio",
            "ic": "insuficiencia cardiaca",
            "icc": "insuficiencia cardiaca congestiva",
            "im": "intramuscular",
            "imc": "indice de massa corporal",
            "ira": "insuficiencia renal aguda",
            "itu": "infeccao do trato urinario",
            "mmii": "membros inferiores",
            "mmss": "membros superiores",
            "pa": "pressao arterial",
            "pc": "perimetro cefalico",
            "pcr": "proteina c reativa",
            "sc": "subcutaneo",
            "sato2": "saturacao de oxigenio",
            "spo2": "saturacao periferica de oxigenio",
            "sv": "sinais vitais",
            "tc": "tomografia computadorizada",
            "tep": "tromboembolismo pulmonar",
            "tfg": "taxa de filtracao glomerular",
            "tvp": "trombose venosa profunda",
            "usg": "ultrassonografia",
            "vo": "via oral",
            "vo2": "consumo de oxigenio",
            "vvp": "acesso venoso periferico",
            "rx": "radiografia",
        }
        self._abbreviation_pattern = self._compile_abbreviation_pattern()
        self.unit_patterns = [
            (re.compile(r"(\d+(?:[.,]\d+)?)\s*mg\b", re.I), r"\1 mg"),
            (re.compile(r"(\d+(?:[.,]\d+)?)\s*ml\b", re.I), r"\1 ml"),
            (re.compile(r"(\d+(?:[.,]\d+)?)\s*mmhg\b", re.I), r"\1 mmHg"),
            (re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:mcg|ug)\b", re.I), r"\1 mcg"),
            (re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:ui|iu)\b", re.I), r"\1 UI"),
            (re.compile(r"(\d+(?:[.,]\d+)?)\s*%", re.I), r"\1%"),
        ]

    def _compile_abbreviation_pattern(self) -> re.Pattern:
        alternatives = sorted(self.abbreviations, key=len, reverse=True)
        return re.compile(
            r"(?<!\w)(" + "|".join(re.escape(item) for item in alternatives) + r")(?!\w)",
            re.IGNORECASE,
        )

    def normalize_abbreviations(self, text: str) -> str:
        """Expande abreviacoes sem remover pontuacao ou alterar palavras maiores."""
        return self._abbreviation_pattern.sub(
            lambda match: self.abbreviations[match.group(0).lower()],
            text,
        )

    def normalize_units(self, text: str) -> str:
        """Padroniza escrita de unidades de medida."""
        for pattern, replacement in self.unit_patterns:
            text = pattern.sub(replacement, text)
        return text

    def normalize(self, text: str) -> str:
        """Aplica todas as normalizacoes."""
        return self.normalize_abbreviations(self.normalize_units(text))

    def add_abbreviation(self, abbr: str, full_form: str) -> None:
        """Permite adicionar novas abreviacoes em runtime."""
        key = abbr.strip().lower()
        if not key or not re.fullmatch(r"\w+", key):
            raise ValueError("A abreviacao deve conter apenas letras, numeros ou underscore.")
        if not full_form.strip():
            raise ValueError("A forma expandida nao pode ser vazia.")
        self.abbreviations[key] = full_form.strip().lower()
        self._abbreviation_pattern = self._compile_abbreviation_pattern()
