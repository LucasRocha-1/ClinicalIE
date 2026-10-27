"""
Extrator baseado em regras (hibrido com ML).
Util como fallback ou para padroes especificos de textos clinicos.
"""

import re
from typing import List, Dict, Any, Pattern


class RuleBasedExtractor:
    """Extrai medidas, frequencias, datas e vias de administracao por regex."""

    def __init__(self):
        self.patterns: Dict[str, Pattern] = {
            "dosage": re.compile(
                r"\b\d+(?:[.,]\d+)?\s*"
                r"(?:mcg\s*/\s*ml|mcg|ug|mg|g|ml|ui|iu)"
                r"(?!\w)",
                re.IGNORECASE,
            ),
            "blood_pressure": re.compile(
                r"\b(?:(?:pa|pressao\s+arterial)\s*[:=]?\s*)?"
                r"\d{2,3}\s*[xX/]\s*\d{2,3}"
                r"(?:\s*mm\s*hg)?\b",
                re.IGNORECASE,
            ),
            "frequency": re.compile(
                r"\b(?:\d+\s*(?:x|vez(?:es)?)\s*(?:/|por)\s*"
                r"(?:dia|d|semana|sem|mes|hora|h)|"
                r"(?:1|2|3|4|6|8|12|24)\s*/\s*d)\b",
                re.IGNORECASE,
            ),
            "date": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
            "heart_rate": re.compile(
                r"\b(?:(?:fc|frequencia\s+cardiaca)\s*[:=]?\s*)?"
                r"\d{2,3}\s*(?:bpm|"
                r"batimentos\s*(?:por|/)\s*min(?:uto)?)\b",
                re.IGNORECASE,
            ),
            "respiratory_rate": re.compile(
                r"\b(?:(?:fr|frequencia\s+respiratoria)\s*[:=]?\s*)?"
                r"\d{1,2}\s*"
                r"(?:irpm|rpm|respira(?:coes|ções)\s*(?:por|/)\s*min(?:uto)?)\b",
                re.IGNORECASE,
            ),
            "oxygen_saturation": re.compile(
                r"\b(?:spo2|sat(?:uracao|uação)?(?:\s*o2)?|"
                r"saturacao(?:\s+periferica)?\s+de\s+oxigenio)\s*[:=]?\s*"
                r"\d{1,3}(?:[.,]\d+)?\s*%?",
                re.IGNORECASE,
            ),
            "temperature": re.compile(
                r"\b(?:temp(?:eratura)?\s*[:=]?\s*)?"
                r"\d{2}(?:[.,]\d+)?\s*°?\s*c\b",
                re.IGNORECASE,
            ),
            "glucose": re.compile(
                r"\b(?:glic(?:emia)?|hgt|hemoglicoteste)\s*[:=]?\s*"
                r"\d{2,3}(?:[.,]\d+)?\s*(?:mg\s*/\s*dl)?\b",
                re.IGNORECASE,
            ),
            "weight": re.compile(
                r"\b(?:peso\s*[:=]?\s*)?\d{1,3}(?:[.,]\d+)?\s*kg\b",
                re.IGNORECASE,
            ),
            "height": re.compile(
                r"\b(?:altura\s*[:=]?\s*)?"
                r"\d(?:[.,]\d{1,2})?\s*m(?:\s*\d{1,2}\s*cm)?\b",
                re.IGNORECASE,
            ),
            "bmi": re.compile(
                r"\b(?:imc\s*[:=]?\s*)?\d{1,2}(?:[.,]\d+)?\s*"
                r"kg\s*/\s*m(?:2|²)\b",
                re.IGNORECASE,
            ),
            "ejection_fraction": re.compile(
                r"\b(?:feve|fra(?:cao|ção)\s+de\s+eje(?:cao|ção))"
                r"\s*[:=]?\s*\d{1,3}(?:[.,]\d+)?\s*%",
                re.IGNORECASE,
            ),
            "duration": re.compile(
                r"\b(?:por\s+)?\d+\s*(?:dias?|semanas?|meses?|anos?|"
                r"horas?|hrs?|h)\b",
                re.IGNORECASE,
            ),
            "route": re.compile(
                r"\b(?:via\s+)?(?:oral|intravenosa|endovenosa|"
                r"endovenoso|intramuscular|subcutanea|subcutaneo|"
                r"sublingual|topica)\b",
                re.IGNORECASE,
            ),
        }

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Aplica todas as regras e retorna matches com offsets de caracteres."""
        results = []

        for label, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                results.append({
                    "text": match.group(0),
                    "label": label.upper(),
                    "start": match.start(),
                    "end": match.end(),
                    "groups": match.groups(),
                    "source": "rule_based",
                })

        return sorted(results, key=lambda entity: (entity["start"], entity["end"]))

    def add_pattern(self, name: str, pattern: str, flags: int = re.IGNORECASE) -> None:
        """Permite adicionar novos padroes dinamicamente."""
        self.patterns[name] = re.compile(pattern, flags)
