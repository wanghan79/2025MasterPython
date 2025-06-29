# -*- coding: utf-8 -*-
from typing import Dict
from models.dialogue_agent import DialogueAgent
from models.scale_agent import ScaleAgent
from models.symptoms_risk_agent import SymptomsRiskAgent
from models.deepseek_client import DeepSeekClient
from models.patent_side_diagnostic import patentDiagnosticAgent
from models.doctor_side_diagnostic import doctorDiagnosticAgent

class Supervisor:
    """
    主控制器类，负责协调对话流程和评估流程
    职责：
    1. 管理会话状态
    2. 协调对话代理和量表代理的工作
    3. 控制整体流程（欢迎->症状分析->量表评估->结果反馈）
    """

    def __init__(self, api_url: str, api_key: str):
        """
        初始化各代理组件
        :param api_url: DeepSeek API地址
        :param api_key: API密钥
        :param scales_directory: 量表数据目录
        """
        self.llm = DeepSeekClient(api_url, api_key)
        self.dialogue_agent = DialogueAgent(self.llm)
        self.scale_agent = ScaleAgent(llm=self.llm)
        self.symptoms_risk_agent = SymptomsRiskAgent(self.llm)
        self.sessions: Dict[str, Dict] = {}  # 存储所有会话数据
        self.patent_side_diagnostic_report = patentDiagnosticAgent(self.llm)
        self.doctor_side_diagnostic_report = doctorDiagnosticAgent(self.llm)

    def _get_session(self, session_id: str) -> Dict:
        """
        获取或创建会话状态
        :param session_id: 会话ID
        :return: 会话状态字典
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "conversation": [],  # 对话历史 (用户输入, 助手回复)
                "assessment": {
                    "active": False,  # 是否在评估中
                    "scale_name": None,  # 当前量表名
                    "progress": 0,  # 当前问题索引
                    "responses": []  # 用户回答记录
                },
                "diagnosis_suspect": None,  # 初步诊断怀疑
                "observed_symptoms": [],  # 观察到的症状
                "user_info": None,  # 用户信息
                "greeting_shown": False,  # 是否已显示欢迎语
                "analysis_phase": False,  # 是否在分析阶段
                "assessment_result": None  # 保存评估结果
            }
        return self.sessions[session_id]

    async def process_input(self, user_input: str, session_id: str) -> Dict:
        """
        处理用户输入的主入口
        :param user_input: 用户输入文本
        :param session_id: 会话ID
        :return: 处理结果字典
        """
        session = self._get_session(session_id)

        # 第一阶段：显示欢迎语
        if not session["greeting_shown"]:
            session["greeting_shown"] = True
            welcome_msg = await self.dialogue_agent.generate_welcome_message(session_id)
            session["conversation"].append(("", welcome_msg))
            return {"type": "dialogue", "content": welcome_msg}

        # 记录用户输入
        session["conversation"].append((user_input, None))

        # 第二阶段：如果在评估流程中
        if session["assessment"]["active"]:
            return await self._handle_assessment_response(user_input, session)

        # 第三阶段：症状分析
        while True:
            analysis = await self.dialogue_agent.analyze_symptoms(
                user_input,
                session["conversation"],
                session["observed_symptoms"]
            )

            session['diagnosis_suspect'] = analysis['likely_condition']
            session["observed_symptoms"].extend(analysis["symptoms_found"])

            if analysis["target_category"] and analysis["likely_condition"] and analysis["should_assess"]:
                return await self._start_assessment(session)
            else:
                return {"type": "dialogue", "content": analysis["next_question"]}

            # 如果模型没有发现有效症状，尝试继续获取更多信息
            if analysis["next_question"]:
                return {"type": "dialogue", "content": analysis["next_question"]}

        # 第四阶段：普通对话处理
        return await self._handle_normal_dialogue(session)

    async def _start_assessment(self, session: Dict) -> Dict:
        """
        开始量表评估流程
        :param session: 会话状态
        :return: 评估开始响应
        """
        scale_name = await self.scale_agent.select_scale(session["diagnosis_suspect"])
        scale = await self.scale_agent.get_scale(scale_name)
        # 设置评估状态
        session["assessment"] = {
            "active": True,
            "scale_name": scale_name,
            "progress": 0,
            "responses": []
        }

        # 返回第一个问题
        first_question = scale["questions"][0]
        context = {
            "current_question": None,  # 第一题还没有当前问题
            "user_input": None,        # 第一题还没有用户输入
            "next_question": first_question
        }
        gentle_question = await self.dialogue_agent._generate_gentle_question(context)
        colored_content = self.dialogue_agent._format_response(gentle_question)

        return {
            "type": "assessment_question",
            "content": colored_content,
            "question": first_question,
            "progress": f"1/{len(scale['questions'])}"
        }

    async def _handle_assessment_response(self, user_input: str, session: Dict) -> Dict:
        assessment = session["assessment"]
        scale = await self.scale_agent.get_scale(assessment["scale_name"])

        current_index = assessment["progress"]
        current_question = scale["questions"][current_index]

        result = await self._process_user_answer(user_input, assessment, current_question)

        # 如果是追问或编号提示，直接返回（未记录答案）
        if result["type"] in ["assessment_follow_up", "assessment_option_selection"]:
            return result

        # 否则已获取有效答案，记录并继续
        answer_idx = result["answer"]
        assessment["responses"].append({
            "question_id": current_question["id"],
            "answer": answer_idx,
            "user_response": user_input
        })
        # 打印当前题目的得分情况
        question_text = current_question.get("question", "（无题干）")
        options = current_question.get("options", [])
        option_text = options[answer_idx] if 0 <= answer_idx < len(options) else "未知选项"
        # print(f"[答题记录] Q{current_index + 1}: {question_text}")
        # print(f"用户选择：{option_text}（得分：{answer_idx}）")
        # 重置状态
        assessment.update({
            "is_follow_up": False,
            "waiting_for_option_selection": False,
            "follow_up_count": 0,
            "progress": current_index + 1
        })

        if assessment["progress"] >= len(scale["questions"]):
            return await self._complete_assessment(session, assessment["scale_name"])
        # 存储当前问题
        current_question = scale["questions"][assessment["progress"] - 1] if assessment["progress"] > 0 else None
        # 生成下一个问题
        next_question = scale["questions"][assessment["progress"]]
        context = {
            "current_question": current_question,
            "user_input": user_input,
            "next_question": next_question
        }
        gentle_question = await self.dialogue_agent._generate_gentle_question(context)

        return {
            "type": "assessment_question",
            "content": self.dialogue_agent._format_response(gentle_question),
            "original_question": next_question,
            "progress": f"{assessment['progress']+1}/{len(scale['questions'])}"
        }

    async def _process_user_answer(self, user_input: str, assessment: Dict, question: Dict) -> Dict:
        if "follow_up_count" not in assessment:
            assessment["follow_up_count"] = 0

        # 判断是否等待用户选择编号
        if assessment.get("waiting_for_option_selection", False):
            try:
                selected_option = int(user_input.strip()) - 1
                if 0 <= selected_option < len(question["options"]):
                    return {"type": "answer", "answer": selected_option}
                else:
                    raise ValueError("编号越界")
            except Exception:
                options_str = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(question["options"])])
                return {
                    "type": "assessment_option_selection",
                    "content": self.dialogue_agent._format_response(
                        f"请输入有效编号（1~{len(question['options'])}）：\n{options_str}"
                    ),
                    "original_question": question
                }

        # 自由回答阶段
        is_follow_up = assessment.get("is_follow_up", False)
        if is_follow_up:
            assessment["follow_up_count"] = assessment.get("follow_up_count", 0) + 1
        follow_up_final = assessment["follow_up_count"] >= 2

        result = await self.scale_agent.parse_answer(
            user_input=user_input,
            question=question,
            is_follow_up=is_follow_up,
            follow_up_count=assessment["follow_up_count"],
            follow_up_final=assessment["follow_up_count"] >= 2
        )

        if isinstance(result, str):
            if result.strip().startswith("为了更准确评估，请直接选择"):
                assessment["waiting_for_option_selection"] = True
                return {
                    "type": "assessment_option_selection",
                    "content": self.dialogue_agent._format_response(result),
                    "original_question": question
                }
            else:
                assessment["is_follow_up"] = True
                return {
                    "type": "assessment_follow_up",
                    "content": self.dialogue_agent._format_response(result),
                    "original_question": question
                }

        return {"type": "answer", "answer": result}



    async def _complete_assessment(self, session: Dict, scale_name: str) -> Dict:
        """
        完成评估并生成结果（增强版：包含症状风险分析）
        """
        responses = session["assessment"]["responses"]

        # 获取量表评估结果
        result = await self.scale_agent.evaluate_responses(
            scale_name,
            responses
        )

        # 获取详细回答记录用于分析
        detailed_responses = []
        scale = await self.scale_agent.get_scale(scale_name)
        for resp in responses:
            qid = resp["question_id"]
            if 0 <= qid - 1 < len(scale["questions"]):
                question = scale["questions"][qid - 1]["question"]
                answer = scale["questions"][qid - 1]["options"][resp["answer"]]
                score = scale["questions"][qid - 1]["scores"][resp["answer"]]
                detailed_responses.append({
                    "question_id": qid,
                    "question": question,
                    "answer": answer,
                    "score": score
                })

        # 进行症状与风险分析
        analysis_result = await self.symptoms_risk_agent.analyze_responses(
            scale_name,
            detailed_responses,
            session["conversation"]
        )

        # 生成患者端诊断报告
        Patient_side_diagnostic_report = await self.patent_side_diagnostic_report.analyze_responses(
            scale_name,
            detailed_responses,
            session["conversation"]
        )

        # 生成医生端端诊断报告
        Doctor_side_diagnostic_report = await self.doctor_side_diagnostic_report.analyze_responses(
            result,
            analysis_result
        )

        # 重置评估状态
        session["assessment"] = {
            "active": False,
            "scale_name": None,
            "progress": 0,
            "responses": []
        }

        # 生成用户友好的总结（包含分析结果）
        summary = await self.dialogue_agent.generate_assessment_summary(
            result,
            analysis_result
        )

        return {
            "type": "assessment_result",
            "content": summary,
            "details": {
                "assessment": result,
                "analysis": analysis_result
            }
        }

    async def _handle_normal_dialogue(self, session: Dict) -> Dict:
        """
        处理普通对话流程
        :param session: 会话状态
        :return: 对话响应
        """
        response = await self.dialogue_agent.continue_conversation(
            session["conversation"],
            session["diagnosis_suspect"]
        )
        return {"type": "dialogue", "content": response}