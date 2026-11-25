"""实例作用域的卡片生命周期，与 Hermes 源码及投递内部变量无关."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from agent.i18n import get_language, t

from .cardkit.builder import build_static_card
from .cardkit.i18n import _t
from .config import Config
from .feishu import FeishuClient
from .streaming.controller import StreamingController
from .streaming.segments import Segment, SegmentType
from .streaming.session import CardSession, SessionState
from .streaming.tooluse import ToolStatus, ToolStep, redact_inline_secrets

_logger = logging.getLogger(__name__)


@dataclass
class Turn:
    session: CardSession
    source_id: str
    text: str = ""
    closing: bool = False
    final_text: str | None = None
    aborted: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    tools: dict[str, ToolStep] = field(default_factory=dict)
    observer_turn_id: str = ""
    usage_ids: set[str] = field(default_factory=set)
    clarify_prompt: Segment | None = None


class StreamCardController(StreamingController):
    """一个适配器持有一个控制器，所有状态只能在所属事件循环访问."""

    def __init__(self, config: Config, client: FeishuClient) -> None:
        self._cfg = config
        self._client = client
        self.turns: dict[str, Turn] = {}
        self._retired: OrderedDict[str, Turn] = OrderedDict()
        self._pending_tools: OrderedDict[int, Turn] = OrderedDict()

    def _cleanup_session(self, session: CardSession) -> None:
        session.flush.mark_completed()
        if session.image_resolver:
            session.image_resolver.cancel_pending()

    async def frame(
        self,
        key: str,
        chat_id: str,
        text: str,
        *,
        reply_to: str | None,
        source_id: str,
        finalize: bool = False,
        session_key: str = "",
    ) -> bool:
        """消费累计正文；只有成功落地才向宿主确认投递."""
        turn = self._retired.get(key) or self.turns.get(key)
        if turn is None:
            session = CardSession(source_id, chat_id, asyncio.get_running_loop())
            session.session_key = session_key
            session.anchor_id = reply_to
            turn = self.turns[key] = Turn(session, source_id)
        async with turn.lock:
            session = turn.session
            if session.chat_id != chat_id:
                return False
            if finalize and session.card_msg_id:
                return await self._deliver_locked(key, turn, text)
            if turn.closing:
                return False
            if session.state == SessionState.IDLE:
                await self._do_create_card(session)
            if session.guard.should_skip("frame") or not session.has_card or session.state == SessionState.FAILED:
                self._retire(key, turn)
                return False
            if finalize:
                return await self._deliver_locked(key, turn, text)
            if text != turn.text:
                self._replace_text(turn, text)
                self._schedule_flush(session)
            return True

    def _turn(self, key: str, chat_id: str) -> Turn | None:
        turn = self.turns.get(key) or self._retired.get(key)
        return turn if turn and turn.session.chat_id == chat_id else None

    def message_id(self, key: str, chat_id: str) -> str | None:
        turn = self._turn(key, chat_id)
        return turn.session.card_msg_id if turn else None

    def source_for_message(self, message_id: str, chat_id: str) -> str | None:
        for turns in (self.turns, self._retired):
            for key, turn in turns.items():
                if turn.session.card_msg_id == message_id and turn.session.chat_id == chat_id:
                    return key
        return None

    async def deliver(self, key: str, chat_id: str, text: str) -> bool:
        turn = self._turn(key, chat_id)
        if not turn:
            return False
        async with turn.lock:
            if turn.aborted or self._client is None:
                return False
            if await self._deliver_locked(key, turn, text):
                return True
            message_id = turn.session.card_msg_id
            if not message_id:
                return False
            # CardKit and IM address different IDs. If the entity update fails,
            # try replacing the contents of the original message under the same lock.
            try:
                await self._client.update_card_message(message_id, build_static_card(text))
            except Exception:
                _logger.warning("Could not update streamed message %s", message_id, exc_info=True)
                return False
            turn.final_text = text
            return True

    async def _deliver_locked(self, key: str, turn: Turn, text: str) -> bool:
        if not turn.session.card_msg_id or turn.aborted:
            return False
        if turn.final_text == text:
            return True
        if text != turn.text:
            is_clarify_prompt = (
                not turn.text and text == t("gateway.clarify.native_stream_placeholder")
                and any(step.name == "clarify" and step.status == ToolStatus.RUNNING for step in turn.tools.values())
            )
            self._replace_text(turn, text)
            if is_clarify_prompt:
                turn.clarify_prompt = turn.session.segment_state.segments[-1]
        if not await self._finish_locked(key, turn):
            return False
        turn.final_text = text
        return True

    async def interim(self, source_id: str, text: str) -> str | None:
        matches = self.find_source(source_id)
        if len(matches) != 1:
            return None
        _, turn = matches[0]
        async with turn.lock:
            if turn.closing or turn.session.guard.should_skip("interim"):
                return None
            # 完整 commentary 可能是刚流过的同一段前奏，不重复追加
            if not text.strip() or not turn.text.rstrip().endswith(text.strip()):
                self._on_thinking_segment(turn.session, "\n\n" + text + "\n\n")
            await turn.session.flush.flush_now(lambda: self._do_flush(turn.session))
            if turn.session.state.is_terminal or turn.session.segment_state.has_dirty:
                return None
            return turn.session.card_msg_id

    def _replace_text(self, turn: Turn, text: str) -> None:
        session = turn.session
        if text.startswith(turn.text):
            session.segment_state.on_answer_delta(text[len(turn.text) :])
        else:
            # 改写已存在的元素；删除本地对象会让远端旧元素残留并重复展示
            for seg in session.active_segments():
                if seg.type == SegmentType.ANSWER:
                    seg.text = ""
                    seg.dirty = True
            if text:
                session.segment_state.on_answer_delta(text)
        turn.text = text

    async def _finish_locked(self, key: str, turn: Turn, *, aborted: bool = False, failed: bool = False) -> bool:
        turn.closing = True
        session = turn.session
        session.footer.setdefault("duration", time.time() - session.created_at)
        if aborted or failed:
            turn.aborted = aborted
            session.state = SessionState.ABORTED if aborted else SessionState.FAILED
        try:
            success = bool(session.card_msg_id) and await self._do_complete_card(session)
        except Exception:
            _logger.exception("Card completion failed")
            success = False
        finally:
            self._retire(key, turn)
        if success and any(s.status == ToolStatus.RUNNING for s in turn.tools.values()):
            self._pending_tools[id(turn)] = turn
            while len(self._pending_tools) > 256:
                self._pending_tools.popitem(last=False)
        return success

    def _retire(self, key: str, turn: Turn) -> None:
        self.turns.pop(key, None)
        self._cleanup_session(turn.session)
        self._retired[key] = turn
        while len(self._retired) > 256:
            self._retired.popitem(last=False)

    def find_source(self, source_id: str) -> list[tuple[str, Turn]]:
        return [(key, turn) for key, turn in self.turns.items() if turn.source_id == source_id and not turn.closing]

    async def observe(self, source_id: str, kind: str, payload: dict[str, Any]) -> None:
        """官方 Hook 补充内容；缺少精确入站身份时不猜测当前聊天回合."""
        if kind == "tool_end":
            call_id = str(payload.get("tool_call_id") or payload.get("call_id") or "")
            observer_id = str(payload.get("turn_id") or "")
            for pending_id, pending in list(self._pending_tools.items()):
                if pending.source_id != source_id or call_id not in pending.tools:
                    continue
                if observer_id and pending.observer_turn_id and observer_id != pending.observer_turn_id:
                    return
                async with pending.lock:
                    if pending.tools[call_id].status == ToolStatus.RUNNING:
                        self._tool_event(pending, kind, payload)
                        await self._do_complete_card(pending.session)
                    if all(s.status != ToolStatus.RUNNING for s in pending.tools.values()):
                        self._pending_tools.pop(pending_id, None)
                return
        matches = self.find_source(source_id)
        if len(matches) != 1:
            return
        key, turn = matches[0]
        async with turn.lock:
            if turn.closing:
                return
            session = turn.session
            observer_id = str(payload.get("turn_id") or "")
            if observer_id:
                if turn.observer_turn_id and turn.observer_turn_id != observer_id:
                    return
                turn.observer_turn_id = observer_id
            if kind == "end":
                if payload.get("interrupted") or payload.get("failed") or not payload.get("completed", True):
                    await self._finish_locked(
                        key,
                        turn,
                        aborted=bool(payload.get("interrupted")),
                        failed=not payload.get("interrupted"),
                    )
                return
            if session.state.is_terminal:
                return
            if kind == "reasoning" and self._cfg.show_reasoning:
                session.segment_state.on_reasoning_delta(str(payload.get("delta", "")))
            elif kind in {"tool_start", "tool_end"}:
                self._tool_event(turn, kind, payload)
            elif kind == "usage":
                session.footer["model"] = str(payload.get("response_model") or payload.get("model") or "")
                request_id = str(payload.get("api_request_id") or "")
                if not request_id or request_id in turn.usage_ids:
                    return
                turn.usage_ids.add(request_id)
                usage = payload.get("usage") or {}
                # 上下文取最近一次完整输入（含缓存），不能使用整回合累计用量
                for name, value in {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "context_length": payload.get("context_length"),
                }.items():
                    session.footer.pop(name, None)
                    if type(value) is int and value >= 0:
                        session.footer[name] = value
                for target, candidates in {
                    "input_tokens": ("input_tokens", "prompt_tokens"),
                    "output_tokens": ("output_tokens", "completion_tokens"),
                }.items():
                    value = next((usage[k] for k in candidates if isinstance(usage.get(k), (int, float))), None)
                    if value is not None:
                        session.footer[target] = session.footer.get(target, 0) + value
            self._schedule_flush(session)

    def _tool_event(self, turn: Turn, kind: str, payload: dict[str, Any]) -> None:
        name = str(payload.get("tool_name", ""))
        call_id = str(payload.get("tool_call_id") or payload.get("call_id") or "")
        if not call_id:
            return
        tracker = turn.session.tool_use
        step = turn.tools.get(call_id)
        new_step = step is None
        if kind == "tool_start":
            if step is not None:
                return
            args = payload.get("args") or {}
            detail = next((str(args[k]) for k in ("command", "path", "query", "url") if args.get(k)), "")
            detail = " ".join(redact_inline_secrets(detail).split())
            step = tracker.record_start(name, detail[:200] + ("…" if len(detail) > 200 else ""))
            if step is not None:
                turn.tools[call_id] = step
        else:
            if step is None:
                step = tracker.record_start(name)
                if step is None:
                    return
                turn.tools[call_id] = step
            if step.status != ToolStatus.RUNNING:
                return
            error = bool(payload.get("is_error")) or payload.get("status") in {"error", "failed"}
            error_text = ""
            if error:
                error_text = redact_inline_secrets(
                    str(payload.get("error_message") or payload.get("result") or "Tool failed")
                )
                error_text = " ".join(error_text.split())
                if len(error_text) > 200:
                    error_text = error_text[:200] + "…"
            # Hook 的完整 result 属于模型上下文，不在卡片重复展示
            tracker.record_end(name, target=step, error=error_text)
            step.elapsed_ms = float(payload.get("duration_ms") or (time.time() - step.started_at) * 1000)
            if step.name == "clarify" and turn.clarify_prompt is not None:
                result = payload.get("result")
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except ValueError:
                        result = None
                answered = (
                    not error and isinstance(result, dict) and result.get("outcome") == "submitted"
                    and any(isinstance(r, dict) and r.get("status") == "answered"
                            for r in (result.get("responses") or []))
                )
                labels = _t("clarify_answered" if answered else "clarify_ended")
                turn.clarify_prompt.text = labels["zh_cn" if get_language().startswith("zh") else "en_us"]
                turn.clarify_prompt.dirty = True
        if not turn.closing:
            segments = turn.session.segment_state
            if new_step:
                segments.on_tool_event(len(turn.tools))
            else:
                # 工具结果可能晚于下一段正文，只刷新所属面板，不另建重复步骤。
                index = list(turn.tools).index(call_id)
                for seg in segments.segments:
                    if (
                        seg.type == SegmentType.TOOL
                        and seg.tool_offset <= index < (seg.tool_end_offset or len(turn.tools))
                    ):
                        seg.dirty = True

    async def close_source(self, source_id: str, *, aborted: bool = True, failed: bool = False) -> bool:
        closed = False
        for key, turn in self.find_source(source_id):
            async with turn.lock:
                if not turn.closing:
                    closed = await self._finish_locked(key, turn, aborted=aborted, failed=failed)
        return closed

    async def close(self) -> None:
        for source_id in {turn.source_id for turn in self.turns.values()}:
            await self.close_source(source_id)
        self._pending_tools.clear()
