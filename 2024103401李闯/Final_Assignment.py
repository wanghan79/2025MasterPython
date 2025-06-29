#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import json
import time
import readline
import requests
import textwrap
from datetime import datetime
from collections import deque
from enum import Enum


# ======================
# 常量定义
# ======================
class Color:
    """ANSI颜色代码"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class CommandType(Enum):
    """命令类型枚举"""
    HELP = 1
    CLEAR = 2
    HISTORY = 3
    CALC = 4
    VERIFY = 5
    TIME = 6
    DATE = 7
    EXIT = 8
    MEMO = 9
    CONFIG = 10
    ABOUT = 11


# ======================
# 配置管理器
# ======================
class ConfigManager:
    """管理应用程序配置"""

    def __init__(self):
        # 此处需要加入我的deepseek api key
        self.EMBEDDED_API_KEY = "< my api key >"

        # API端点配置
        self.API_URL = "https://api.deepseek.com/v1/chat/completions"
        self.MODEL = "deepseek-reasoner"
        self.MAX_TOKENS = 4096
        self.TEMPERATURE = 0.7

        # 应用程序设置
        self.MAX_HISTORY = 10
        self.USER_NAME = "用户"
        self.BOT_NAME = "DeepSeek助手"
        self.WELCOME_MESSAGE = (
            f"{Color.BOLD}{Color.BLUE}=== 欢迎使用DeepSeek聊天机器人 ==={Color.END}\n"
            f"我是{self.BOT_NAME}，可以回答您的问题、进行数学计算或验证计算结果\n"
            f"输入 {Color.GREEN}/help{Color.END} 查看可用命令\n"
            f"输入 {Color.GREEN}/exit{Color.END} 退出程序"
        )

        # 系统提示词
        self.SYSTEM_PROMPT = (
                "你是一个专业、友好且知识渊博的AI助手，尤其擅长数学和逻辑推理。"
                "当用户提出数学问题时，你会提供准确的计算和解释。"
                "如果用户要求验证计算，你会仔细检查并提供反馈。"
                "你的回答应简洁、专业且富有帮助性。"
                "当前日期: " + datetime.now().strftime("%Y年%m月%d日 %A")
        )

        # 备忘录存储
        self.memos = []
        self.memo_file = "deepseek_memos.json"
        self.load_memos()

    def load_memos(self):
        """从文件加载备忘录"""
        if os.path.exists(self.memo_file):
            try:
                with open(self.memo_file, 'r', encoding='utf-8') as f:
                    self.memos = json.load(f)
            except:
                self.memos = []

    def save_memos(self):
        """保存备忘录到文件"""
        try:
            with open(self.memo_file, 'w', encoding='utf-8') as f:
                json.dump(self.memos, f, ensure_ascii=False, indent=2)
            return True
        except:
            return False


# ======================
# 数学引擎
# ======================
class MathEngine:
    """执行数学计算和验证的引擎"""

    def __init__(self):
        # 支持的数学函数和常量
        self.supported_functions = {
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'asin': math.asin,
            'acos': math.acos,
            'atan': math.atan,
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,
            'sqrt': math.sqrt,
            'log': math.log10,
            'ln': math.log,
            'log2': math.log2,
            'exp': math.exp,
            'abs': abs,
            'ceil': math.ceil,
            'floor': math.floor,
            'round': round,
            'pi': math.pi,
            'e': math.e,
            'tau': math.tau,
            'inf': math.inf,
            'deg': math.degrees,
            'rad': math.radians
        }

        # 运算符优先级
        self.operator_precedence = {
            '+': 1,
            '-': 1,
            '*': 2,
            '/': 2,
            '%': 2,
            '^': 3,
            '**': 3
        }

    def tokenize_expression(self, expression):
        """将表达式拆分为标记"""
        tokens = []
        current_token = ""

        for char in expression:
            if char.isspace():
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
            elif char in '()+-*/%^.,':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            else:
                current_token += char

        if current_token:
            tokens.append(current_token)

        return tokens

    def shunting_yard(self, tokens):
        """使用调度场算法将中缀表达式转换为后缀表达式"""
        output = []
        operators = []

        for token in tokens:
            if token.replace('.', '', 1).isdigit() or token.replace('.', '', 1).isnumeric():
                output.append(float(token) if '.' in token else int(token))
            elif token in self.supported_functions:
                operators.append(token)
            elif token in self.operator_precedence:
                while operators and operators[-1] != '(' and (
                        operators[-1] in self.supported_functions or
                        self.operator_precedence[operators[-1]] > self.operator_precedence[token] or
                        (self.operator_precedence[operators[-1]] == self.operator_precedence[token] and token != '^')):
                    output.append(operators.pop())
                operators.append(token)
            elif token == '(':
                operators.append(token)
            elif token == ')':
                while operators and operators[-1] != '(':
                    output.append(operators.pop())
                if operators and operators[-1] == '(':
                    operators.pop()
                if operators and operators[-1] in self.supported_functions:
                    output.append(operators.pop())

        while operators:
            output.append(operators.pop())

        return output

    def evaluate_rpn(self, rpn):
        """评估后缀表达式"""
        stack = []

        for token in rpn:
            if isinstance(token, (int, float)):
                stack.append(token)
            elif token in self.operator_precedence:
                if len(stack) < 2:
                    raise ValueError("无效表达式：运算符缺少操作数")
                b = stack.pop()
                a = stack.pop()

                if token == '+':
                    stack.append(a + b)
                elif token == '-':
                    stack.append(a - b)
                elif token == '*':
                    stack.append(a * b)
                elif token == '/':
                    if b == 0:
                        raise ValueError("数学错误：除以零")
                    stack.append(a / b)
                elif token == '%':
                    stack.append(a % b)
                elif token in ('^', '**'):
                    stack.append(a ** b)
            elif token in self.supported_functions:
                if callable(self.supported_functions[token]):
                    if len(stack) < 1:
                        raise ValueError(f"函数 '{token}' 缺少参数")
                    arg = stack.pop()
                    result = self.supported_functions[token](arg)
                    stack.append(result)
                else:
                    stack.append(self.supported_functions[token])
            else:
                raise ValueError(f"未知的符号或函数: '{token}'")

        if len(stack) != 1:
            raise ValueError("无效的表达式")

        return stack[0]

    def evaluate_expression(self, expression):
        """安全地评估数学表达式"""
        try:
            # 预处理表达式
            expression = expression.replace(' ', '')
            expression = expression.replace('**', '^')
            expression = expression.lower()

            # 将表达式拆分为标记
            tokens = self.tokenize_expression(expression)

            # 转换为后缀表达式
            rpn = self.shunting_yard(tokens)

            # 评估后缀表达式
            result = self.evaluate_rpn(rpn)

            # 处理浮点数精度问题
            if isinstance(result, float) and result.is_integer():
                result = int(result)

            return result
        except Exception as e:
            raise ValueError(f"计算错误: {str(e)}")

    def verify_calculation(self, statement):
        """验证用户提供的计算语句是否正确"""
        try:
            # 提取表达式和结果
            match = re.match(r'(.+?)(?:=|等于|是)\s*([-+]?\d*\.?\d+)', statement)
            if not match:
                return None, "格式错误。请使用类似'1+1=2'的格式"

            expression, user_result = match.groups()
            expression = expression.strip()
            user_result = float(user_result.strip())

            # 计算实际结果
            actual_result = self.evaluate_expression(expression)

            # 比较结果
            tolerance = 1e-10
            if abs(actual_result - user_result) < tolerance:
                return True, f"正确！{expression} = {actual_result}"
            else:
                return False, f"不正确。{expression} 的正确结果是 {actual_result}，而不是 {user_result}"
        except Exception as e:
            return None, str(e)


# ======================
# API 客户端
# ======================
class DeepSeekAPIClient:
    """处理与DeepSeek API的通信"""

    def __init__(self, config):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.EMBEDDED_API_KEY}",
            "Content-Type": "application/json"
        }
        self.request_count = 0
        self.start_time = time.time()

    def send_request(self, messages):
        """发送请求到DeepSeek API"""
        payload = {
            "model": self.config.MODEL,
            "messages": messages,
            "max_tokens": self.config.MAX_TOKENS,
            "temperature": self.config.TEMPERATURE,
            "stream": False
        }

        try:
            start_time = time.time()
            response = requests.post(
                self.config.API_URL,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response_time = time.time() - start_time

            self.request_count += 1

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                usage = result.get('usage', {})

                # 添加API统计信息
                stats = (
                    f"\n{Color.YELLOW}[API统计] "
                    f"耗时: {response_time:.2f}s | "
                    f"Tokens: {usage.get('prompt_tokens', 0)}/{usage.get('completion_tokens', 0)} | "
                    f"总请求: {self.request_count}{Color.END}"
                )

                return content + stats, None
            else:
                error_msg = f"API错误: {response.status_code}"
                if response.text:
                    try:
                        error_data = response.json()
                        error_msg += f" - {error_data.get('error', {}).get('message', '未知错误')}"
                    except:
                        error_msg += f" - {response.text[:200]}"
                return None, error_msg
        except requests.exceptions.RequestException as e:
            return None, f"网络错误: {str(e)}"


# ======================
# 聊天历史管理器
# ======================
class ChatHistoryManager:
    """管理对话历史记录"""

    def __init__(self, config):
        self.config = config
        self.history = deque(maxlen=config.MAX_HISTORY)
        self.system_message = {"role": "system", "content": config.SYSTEM_PROMPT}

        # 添加初始系统消息
        self.add_message("system", config.SYSTEM_PROMPT)

    def add_message(self, role, content):
        """添加消息到历史记录"""
        self.history.append({"role": role, "content": content})

    def get_history(self):
        """获取当前历史记录（包括系统消息）"""
        return [self.system_message] + list(self.history)

    def clear_history(self):
        """清除历史记录（保留系统消息）"""
        self.history.clear()

    def get_formatted_history(self):
        """获取格式化的历史记录（用于显示）"""
        if not self.history:
            return "对话历史为空"

        history_text = f"{Color.CYAN}对话历史:{Color.END}\n"
        for i, msg in enumerate(self.history):
            prefix = f"{self.config.USER_NAME}: " if msg["role"] == "user" else f"{self.config.BOT_NAME}: "
            color = Color.GREEN if msg["role"] == "user" else Color.BLUE
            history_text += f"{i + 1}. {color}{prefix}{msg['content']}{Color.END}\n"
        return history_text


# ======================
# 命令处理器
# ======================
class CommandHandler:
    """处理用户命令和特殊操作"""

    def __init__(self, config, math_engine, chat_history):
        self.config = config
        self.math_engine = math_engine
        self.chat_history = chat_history

        # 命令映射
        self.command_map = {
            "help": CommandType.HELP,
            "clear": CommandType.CLEAR,
            "history": CommandType.HISTORY,
            "calc": CommandType.CALC,
            "calculate": CommandType.CALC,
            "verify": CommandType.VERIFY,
            "time": CommandType.TIME,
            "date": CommandType.DATE,
            "exit": CommandType.EXIT,
            "quit": CommandType.EXIT,
            "memo": CommandType.MEMO,
            "note": CommandType.MEMO,
            "config": CommandType.CONFIG,
            "about": CommandType.ABOUT,
            "info": CommandType.ABOUT
        }

    def parse_command(self, user_input):
        """解析用户输入的命令"""
        if not user_input.startswith('/'):
            return None, None

        parts = user_input[1:].split(' ', 1)
        command_key = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        command_type = self.command_map.get(command_key)
        return command_type, args

    def execute_command(self, command_type, args):
        """执行命令"""
        if command_type == CommandType.HELP:
            return self.show_help()
        elif command_type == CommandType.CLEAR:
            return self.clear_history()
        elif command_type == CommandType.HISTORY:
            return self.show_history()
        elif command_type == CommandType.CALC:
            return self.perform_calculation(args)
        elif command_type == CommandType.VERIFY:
            return self.verify_calculation(args)
        elif command_type == CommandType.TIME:
            return self.show_time()
        elif command_type == CommandType.DATE:
            return self.show_date()
        elif command_type == CommandType.MEMO:
            return self.handle_memo(args)
        elif command_type == CommandType.CONFIG:
            return self.show_config()
        elif command_type == CommandType.ABOUT:
            return self.show_about()
        elif command_type == CommandType.EXIT:
            return "exit"
        else:
            return f"未知命令。输入 {Color.GREEN}/help{Color.END} 查看可用命令"

    def show_help(self):
        """显示帮助信息"""
        help_text = f"""
{Color.BOLD}{Color.UNDERLINE}可用命令:{Color.END}

{Color.GREEN}/help{Color.END} - 显示此帮助信息
{Color.GREEN}/clear{Color.END} - 清除对话历史
{Color.GREEN}/history{Color.END} - 显示对话历史
{Color.GREEN}/calc [表达式]{Color.END} - 执行数学计算
{Color.GREEN}/verify [表达式]=[结果]{Color.END} - 验证计算是否正确
{Color.GREEN}/time{Color.END} - 显示当前时间
{Color.GREEN}/date{Color.END} - 显示当前日期
{Color.GREEN}/memo [内容]{Color.END} - 添加备忘录 (不加内容查看所有备忘录)
{Color.GREEN}/config{Color.END} - 显示当前配置
{Color.GREEN}/about{Color.END} - 显示程序信息
{Color.GREEN}/exit{Color.END} - 退出程序

{Color.BOLD}{Color.UNDERLINE}示例:{Color.END}
{Color.YELLOW}/calc 2 + 3 * (4 - 1)
/verify 2^3 + 4/2=10
/memo 记得明天下午开会
/memo list
/memo delete 1{Color.END}
"""
        return help_text.strip()

    def clear_history(self):
        """清除对话历史"""
        self.chat_history.clear_history()
        return "对话历史已清除"

    def show_history(self):
        """显示对话历史"""
        return self.chat_history.get_formatted_history()

    def perform_calculation(self, expression):
        """执行数学计算"""
        if not expression:
            return f"请提供要计算的表达式。用法: {Color.YELLOW}/calc 2 + 3 * 4{Color.END}"

        try:
            result = self.math_engine.evaluate_expression(expression)
            return f"{Color.BOLD}计算结果:{Color.END} {expression} = {Color.GREEN}{result}{Color.END}"
        except Exception as e:
            return f"{Color.RED}{str(e)}{Color.END}"

    def verify_calculation(self, statement):
        """验证计算是否正确"""
        if not statement:
            return f"请提供要验证的表达式。用法: {Color.YELLOW}/verify 2^3 + 4/2=10{Color.END}"

        success, message = self.math_engine.verify_calculation(statement)
        if success is True:
            return f"{Color.GREEN}{message}{Color.END}"
        elif success is False:
            return f"{Color.RED}{message}{Color.END}"
        else:
            return f"{Color.YELLOW}{message}{Color.END}"

    def show_time(self):
        """显示当前时间"""
        current_time = datetime.now().strftime("%H:%M:%S")
        return f"当前时间: {Color.GREEN}{current_time}{Color.END}"

    def show_date(self):
        """显示当前日期"""
        current_date = datetime.now().strftime("%Y年%m月%d日 %A")
        return f"当前日期: {Color.GREEN}{current_date}{Color.END}"

    def handle_memo(self, args):
        """处理备忘录命令"""
        if not args:
            # 显示所有备忘录
            if not self.config.memos:
                return "没有备忘录"

            memo_text = f"{Color.CYAN}备忘录列表:{Color.END}\n"
            for i, memo in enumerate(self.config.memos, 1):
                memo_text += f"{i}. {memo}\n"
            return memo_text

        if args.lower() == "list":
            return self.handle_memo("")

        if args.lower().startswith("delete"):
            parts = args.split(' ', 1)
            if len(parts) < 2:
                return f"用法: {Color.YELLOW}/memo delete [序号]{Color.END}"

            try:
                index = int(parts[1]) - 1
                if 0 <= index < len(self.config.memos):
                    removed = self.config.memos.pop(index)
                    self.config.save_memos()
                    return f"已删除备忘录: {Color.RED}{removed}{Color.END}"
                else:
                    return f"{Color.RED}无效的备忘录序号{Color.END}"
            except ValueError:
                return f"{Color.RED}无效的序号格式{Color.END}"

        # 添加新备忘录
        self.config.memos.append(args)
        self.config.save_memos()
        return f"已添加备忘录: {Color.GREEN}{args}{Color.END}"

    def show_config(self):
        """显示当前配置"""
        config_text = f"""
{Color.BOLD}{Color.UNDERLINE}当前配置:{Color.END}

{Color.CYAN}模型:{Color.END} {self.config.MODEL}
{Color.CYAN}最大历史记录:{Color.END} {self.config.MAX_HISTORY}
{Color.CYAN}最大Tokens:{Color.END} {self.config.MAX_TOKENS}
{Color.CYAN}温度:{Color.END} {self.config.TEMPERATURE}
{Color.CYAN}用户名称:{Color.END} {self.config.USER_NAME}
{Color.CYAN}助手名称:{Color.END} {self.config.BOT_NAME}
{Color.CYAN}备忘录数量:{Color.END} {len(self.config.memos)}
"""
        return config_text.strip()

    def show_about(self):
        """显示程序信息"""
        about_text = f"""
{Color.BOLD}{Color.BLUE}DeepSeek聊天机器人 - 嵌入式API版本{Color.END}

{Color.UNDERLINE}功能:{Color.END}
- 自然语言对话
- 数学计算和验证
- 对话历史记录
- 备忘录功能
- 多种实用命令

{Color.UNDERLINE}技术支持:{Color.END}
- 使用DeepSeek API提供AI能力
- 自定义数学引擎
- 命令行界面优化

{Color.YELLOW}提示: 本程序已集成API密钥，无需用户配置{Color.END}
"""
        return about_text.strip()


# ======================
# 主聊天机器人
# ======================
class DeepSeekChatbot:
    """DeepSeek聊天机器人主类"""

    def __init__(self):
        self.config = ConfigManager()
        self.math_engine = MathEngine()
        self.chat_history = ChatHistoryManager(self.config)
        self.api_client = DeepSeekAPIClient(self.config)
        self.command_handler = CommandHandler(
            self.config, self.math_engine, self.chat_history
        )
        self.running = True

    def display_welcome(self):
        """显示欢迎信息"""
        print(self.config.WELCOME_MESSAGE)
        print(f"{Color.CYAN}当前日期: {datetime.now().strftime('%Y年%m月%d日 %A')}{Color.END}")
        print(f"{Color.CYAN}系统提示: {textwrap.shorten(self.config.SYSTEM_PROMPT, 100)}{Color.END}")

    def start(self):
        """启动聊天机器人"""
        self.display_welcome()

        while self.running:
            try:
                # 获取用户输入
                user_input = input(f"{Color.GREEN}{self.config.USER_NAME}>{Color.END} ").strip()

                # 处理空输入
                if not user_input:
                    continue

                # 检查是否为命令
                command_type, args = self.command_handler.parse_command(user_input)
                if command_type is not None:
                    response = self.command_handler.execute_command(command_type, args)
                    if response == "exit":
                        self.running = False
                    elif response:
                        print(f"{Color.BLUE}{self.config.BOT_NAME}>{Color.END} {response}")
                    continue

                # 添加到历史记录
                self.chat_history.add_message("user", user_input)

                # 检测并处理数学问题
                math_response = self.detect_and_handle_math(user_input)
                if math_response:
                    print(f"{Color.BLUE}{self.config.BOT_NAME}>{Color.END} {math_response}")
                    self.chat_history.add_message("assistant", math_response)
                    continue

                # 发送到DeepSeek API
                response, error = self.api_client.send_request(self.chat_history.get_history())

                if error:
                    print(f"{Color.RED}{self.config.BOT_NAME}> [错误] {error}{Color.END}")
                else:
                    print(f"{Color.BLUE}{self.config.BOT_NAME}>{Color.END} {response}")
                    self.chat_history.add_message("assistant", response)

            except KeyboardInterrupt:
                print(f"\n{Color.YELLOW}提示: 使用 {Color.GREEN}/exit{Color.YELLOW} 退出程序{Color.END}")
            except Exception as e:
                print(f"{Color.RED}{self.config.BOT_NAME}> [错误] 发生错误: {str(e)}{Color.END}")

    def detect_and_handle_math(self, user_input):
        """检测并处理数学问题"""
        # 检测计算验证请求
        verify_pattern = r'(.+?)(?:=|等于|是)\s*([-+]?\d*\.?\d+)\s*\?$'
        verify_match = re.match(verify_pattern, user_input)
        if verify_match:
            expression = verify_match.group(1).strip()
            user_result = verify_match.group(2).strip()
            statement = f"{expression}={user_result}"
            success, message = self.math_engine.verify_calculation(statement)
            if success is True:
                return f"{Color.GREEN}[计算验证] {message}{Color.END}"
            elif success is False:
                return f"{Color.RED}[计算验证] {message}{Color.END}"
            else:
                return f"{Color.YELLOW}[计算验证] {message}{Color.END}"

        # 检测直接计算请求
        calc_pattern = r'^(计算|计算:|计算：|求解|计算一下|请计算)\s*(.+)$'
        calc_match = re.match(calc_pattern, user_input)
        if calc_match:
            expression = calc_match.group(2).strip()
            if expression:
                try:
                    result = self.math_engine.evaluate_expression(expression)
                    return f"{Color.GREEN}[计算模式] {expression} = {result}{Color.END}"
                except Exception as e:
                    return f"{Color.RED}[计算模式] {str(e)}{Color.END}"

        # 检测简单数学表达式
        math_pattern = r'^[\d\s\.\(\)\+\-\*\/\^\%!a-z]+$'
        if re.match(math_pattern, user_input):
            try:
                result = self.math_engine.evaluate_expression(user_input)
                return f"{Color.GREEN}[计算模式] {user_input} = {result}{Color.END}"
            except:
                pass

        return None

    def shutdown(self):
        """关闭程序前执行清理"""
        print(f"{Color.BLUE}{self.config.BOT_NAME}>{Color.END} 正在保存数据...")
        self.config.save_memos()
        print(f"{Color.BLUE}{self.config.BOT_NAME}>{Color.END} 再见！")


# ======================
# 程序入口
# ======================
if __name__ == "__main__":
    # 创建并启动聊天机器人
    bot = DeepSeekChatbot()

    try:
        # 设置命令行历史记录
        try:
            readline.read_history_file('.deepseek_history')
        except FileNotFoundError:
            pass

        # 启动聊天机器人
        bot.start()

        # 关闭程序
        bot.shutdown()

        # 保存命令行历史记录
        try:
            readline.write_history_file('.deepseek_history')
        except Exception:
            pass

        print(f"{Color.BOLD}{Color.BLUE}=== 聊天机器人已退出 ==={Color.END}")
    except Exception as e:
        print(f"{Color.RED}发生严重错误: {str(e)}{Color.END}")