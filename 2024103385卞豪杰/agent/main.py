# -*- coding: utf-8 -*-
import asyncio
import argparse
from models.supervisor import Supervisor


async def interactive_chat(api_url, api_key):
    print("\n=== 心理健康助手 ===")
    print("输入 '退出' 结束对话\n")

    supervisor = Supervisor(api_url, api_key)
    session_id = input("请输入您的姓名或ID: ").strip() or "default_user"

    # 第一次处理（欢迎消息）
    response = await supervisor.process_input("", session_id)
    print(f"\n助手: {response['content']}")

    #整体项目的入口


if __name__ == "__main__":
    parser = argparse.ArgumentParser()