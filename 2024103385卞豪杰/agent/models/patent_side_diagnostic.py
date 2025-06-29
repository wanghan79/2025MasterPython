# -*- coding: utf-8 -*-
import re
from typing import Dict, List, Tuple



class patentDiagnosticAgent:

    async def analyze_responses(self, scale_name: str, responses: List[Dict],
                                conversation: List[Tuple[str, str]]) -> None:
        output = ""
        print("\n=== 患者端诊断报告 ===\n")
        print(output)