import json
import os
from collections.abc import AsyncGenerator

from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langsmith import traceable

from app.agent.agent_middleware import get_middleware
from app.agent.agent_tools import (
    create_note_tool,
    get_note_stats_tool,
    get_related_notes_tool,
    get_today_reviews_tool,
    get_user_info_tools,
    mark_reviewed_tool,
    rag_summary_tools,
    search_notes_tool,
    set_current_user_id,
    what_time_is_now,
)
from app.core.logger_handler import logger
from app.services import session_manager as sm
from app.utils.prompt_loader import load_prompt


class AgentFactory:
    """
    Agent 工厂类（LangGraph 版本）
    支持：
    - 每次调用创建全新的 LangGraph Agent 实例
    - 动态注入工具、提示词、模型配置
    - 支持异步流式调用
    """

    def __init__(
            self,
            model: str = "qwen3-max",
            api_key: str | None = None,
            default_tools: list[BaseTool] | None = None,
            default_middleware: list | None = None,
            default_system_prompt: str | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("CHAT_API_KEY")
        self.default_tools = default_tools or self._get_default_tools()
        self.default_middleware = default_middleware or self._get_default_middleware()
        self.default_system_prompt = default_system_prompt or self._get_default_system_prompt()

    @staticmethod
    def _get_default_tools() -> list[BaseTool]:
        """获取默认工具列表"""
        return [
            rag_summary_tools,
            what_time_is_now,
            get_user_info_tools,
            search_notes_tool,
            get_note_stats_tool,
            get_today_reviews_tool,
            mark_reviewed_tool,
            create_note_tool,
            get_related_notes_tool,
        ]

    def _get_default_middleware(self) -> list:
        """获取默认中间件列表（预留，未来接入）"""
        return get_middleware()

    @staticmethod
    def _get_default_system_prompt() -> str:
        """获取默认系统提示词"""
        return load_prompt('main_prompt')

    def _create_chat_model(self, custom_model: str | None = None):
        """根据 LLM_TYPE 创建聊天模型实例"""
        llm_type = os.getenv("LLM_TYPE", "ALIYUN").upper()

        if llm_type == "OLLAMA":
            model_name = custom_model or os.getenv("OLLAMA_MODEL_NAME", self.model)
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            logger.info(f"🤖 Agent使用Ollama模型: {model_name}")
            return ChatOllama(
                model=model_name,
                base_url=base_url,
                streaming=True,
                top_p=0.7,
            )

        elif llm_type == "ALIYUN":
            api_key = os.getenv("ALIYUN_ACCESS_KEY_SECRET")
            base_url = os.getenv("ALIYUN_BASE_URL")
            model_name = custom_model or os.getenv("ALIYUN_MODEL_NAME", self.model)
            logger.info(f"🤖 Agent使用阿里云百炼模型: {model_name}")
            return ChatTongyi(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                streaming=True,
                top_p=0.7,
            )

        else:
            raise ValueError(f"不支持的LLM_TYPE: {llm_type}，可选值: ALIYUN, OLLAMA")

    def create_agent(
            self,
            custom_tools: list[BaseTool] | None = None,
            custom_model: str | None = None,
    ):
        """
        核心工厂方法：创建 LangGraph ReAct Agent
        每次调用生成新实例，避免全局状态污染
        """
        chat_model = self._create_chat_model(custom_model)
        tools = custom_tools or self.default_tools

        return create_agent(
            chat_model,
            tools,
            system_prompt=self.default_system_prompt,
        )


# 初始化全局工厂
agent_factory = AgentFactory()


def _build_messages(history: list[tuple] | None, query: str) -> list[BaseMessage]:
    """将会话历史 + 当前查询构建为 LangGraph 消息列表"""
    messages: list[BaseMessage] = []
    if history:
        for user_msg, assistant_msg in history:
            messages.append(HumanMessage(content=user_msg))
            messages.append(AIMessage(content=assistant_msg))
    messages.append(HumanMessage(content=query))
    return messages


def _extract_steps_from_messages(messages: list[BaseMessage]) -> list[dict]:
    """从 LangGraph 返回的 messages 中提取工具调用步骤"""
    steps = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                steps.append({
                    "tool": tc["name"],
                    "tool_input": tc["args"],
                    "tool_output": None,  # 将在下面匹配 ToolMessage 时填充
                })
        elif isinstance(msg, ToolMessage):
            # 回填最后一个匹配 tool_call_id 的步骤
            for step in reversed(steps):
                if step["tool_output"] is None:
                    step["tool_output"] = msg.content
                    break
    return steps


async def get_agent_response(
        query: str,
        history: list[tuple] | None = None,
        user_id: str | None = None,
        custom_tools: list[BaseTool] | None = None,
        **kwargs
):
    """
    获取 Agent 响应（非流式版本）
    :param query: 用户查询
    :param history: 会话历史 [(user_msg, assistant_msg), ...]
    :param user_id: 用户ID
    :param custom_tools: 自定义工具（可选）
    :return: {"response": str, "steps": list}
    """
    if user_id:
        set_current_user_id(user_id)

    try:
        agent = agent_factory.create_agent(custom_tools=custom_tools)
        messages = _build_messages(history, query)

        result = await agent.ainvoke({"messages": messages})

        # 从返回的 messages 中提取最终回复和中间步骤
        response_messages = result["messages"]
        steps = _extract_steps_from_messages(response_messages)

        # 最终回复是最后一条 AIMessage（非工具调用）
        final_response = ""
        for msg in reversed(response_messages):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                final_response = msg.content
                break

        return {
            "response": final_response or "抱歉，我无法理解您的请求。",
            "steps": steps
        }

    except Exception as e:
        logger.error(f"Agent 执行错误: {str(e)}", exc_info=True)
        return {
            "response": f"抱歉，处理您的请求时出现了错误: {str(e)}",
            "steps": []
        }


@traceable
async def get_agent_stream_response(
        query: str,
        session_id: str,
        user_id: str,
        custom_tools: list[BaseTool] | None = None,
        **kwargs
) -> AsyncGenerator[str, None]:
    """
    获取 Agent 流式响应（真正逐 token 流式 + 工具调用事件）
    :param query: 用户查询
    :param session_id: 会话 ID
    :param user_id: 用户 ID
    :param custom_tools: 自定义工具（可选）
    :return: SSE 事件流
    """
    set_current_user_id(user_id)

    try:
        logger.info(f"【Agent流式响应】开始处理请求，用户ID: {user_id}, 会话ID: {session_id}, 查询: {query}")

        # 获取会话历史
        history = await sm.session_manager.get_history(session_id, user_id)
        logger.info(f"【Agent流式响应】获取会话历史成功，历史记录数: {len(history)}")

        agent = agent_factory.create_agent(custom_tools=custom_tools)
        messages = _build_messages(history, query)

        # 收集完整响应用于存储到会话历史
        full_response = []
        # 每轮 LLM 调用的 token 缓冲区，用于区分"思考"和"最终回答"
        current_llm_buffer = []
        has_tool_calls_in_current_turn = False

        async for event in agent.astream_events({"messages": messages}, version="v2"):
            event_kind = event["event"]

            # 只处理来自 Agent 节点的 LLM 事件，忽略工具内部（node=tools）的嵌套 LLM 调用
            langgraph_node = event.get("metadata", {}).get("langgraph_node", "")
            is_agent_llm = langgraph_node == "model"

            if event_kind == "on_chat_model_start" and is_agent_llm:
                # 新一轮 Agent LLM 调用开始，重置缓冲区
                current_llm_buffer = []
                has_tool_calls_in_current_turn = False

            elif event_kind == "on_chat_model_stream" and is_agent_llm:
                chunk = event["data"]["chunk"]
                # 检测是否包含工具调用（说明这轮是"思考→决定调工具"）
                if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                    has_tool_calls_in_current_turn = True
                token = chunk.content if hasattr(chunk, "content") else ""
                if token:
                    current_llm_buffer.append(token)

            elif event_kind == "on_chat_model_end" and is_agent_llm:
                # 一轮 Agent LLM 调用结束，根据是否有工具调用决定事件类型
                content = "".join(current_llm_buffer)
                if content:
                    if has_tool_calls_in_current_turn:
                        # 这是中间思考（模型决定调用工具前的推理）
                        logger.info(f"🧠 [Agent 思考] {content[:200]}")
                        yield f"data: {json.dumps({'type': 'thinking', 'stage': 'reasoning', 'content': content}, ensure_ascii=False)}\n\n"
                    else:
                        # 这是最终回答（不再调用工具）
                        full_response.append(content)
                        # 分块推送，模拟打字机效果（与旧版行为一致）
                        chunk_size = 15
                        for i in range(0, len(content), chunk_size):
                            chunk_text = content[i:i + chunk_size]
                            yield f"data: {json.dumps({'type': 'response', 'content': chunk_text}, ensure_ascii=False)}\n\n"
                current_llm_buffer = []
                has_tool_calls_in_current_turn = False

            elif event_kind == "on_tool_start":
                # 工具调用开始
                tool_name = event.get("name", "unknown")
                tool_input = event["data"].get("input", {})
                logger.info(f"🛠️ [调用工具] {tool_name}，输入: {tool_input}")
                yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name, 'input': tool_input}, ensure_ascii=False)}\n\n"

            elif event_kind == "on_tool_end":
                # 工具调用结束
                tool_name = event.get("name", "unknown")
                tool_output = event["data"].get("output", "")
                output_preview = str(tool_output)[:200] + "..." if len(str(tool_output)) > 200 else str(tool_output)
                logger.info(f"📤 [工具结果] {tool_name}: {output_preview}")
                yield f"data: {json.dumps({'type': 'tool_end', 'tool': tool_name, 'output': output_preview}, ensure_ascii=False)}\n\n"

            elif event_kind == "on_custom_event" and event.get("name") == "rag_progress":
                # RAG 内部细粒度进度事件
                progress_data = event["data"]
                logger.info(f"💭 [RAG进度] {progress_data.get('stage', 'unknown')}: {progress_data.get('content', '')}")
                yield f"data: {json.dumps({'type': 'thinking', **progress_data}, ensure_ascii=False)}\n\n"

        # 保存到会话历史
        response_text = "".join(full_response)
        if response_text:
            await sm.session_manager.add_message(session_id, user_id, query, response_text)
            logger.info("【Agent流式响应】添加到会话历史成功")

        # 发送结束标记
        yield f"data: {json.dumps({'type': 'done', 'session_id': session_id}, ensure_ascii=False)}\n\n"
        logger.info(f"【Agent流式响应】处理完成，会话ID: {session_id}")

    except Exception as e:
        logger.error(f"【Agent流式响应】处理请求失败: {e}", exc_info=True)
        error_message = f"错误: {str(e)}"
        yield f"data: {json.dumps({'type': 'error', 'content': error_message, 'session_id': session_id}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
