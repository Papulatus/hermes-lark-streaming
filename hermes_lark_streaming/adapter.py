"""继承内置连接与权限，仅替换出站卡片展示."""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections import OrderedDict, deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from sys import maxsize
from typing import Any, cast

from gateway import session_context
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, ProcessingOutcome, SendResult
from gateway.session_context import get_session_env
from plugins.platforms.feishu import adapter as builtin

from .cardkit.builder import build_static_card
from .config import DEFAULT_DOMAIN, LARK_DOMAIN, Config, hermes_home
from .controller import StreamCardController
from .feishu import FeishuClient, FeishuClientConfig

_logger = logging.getLogger(__name__)
ACTIVE: weakref.WeakSet[Any] = weakref.WeakSet()


@dataclass
class CardObservations:
    pending: deque[tuple[str, dict[str, Any]]] = field(default_factory=lambda: deque(maxlen=256))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    streaming: bool = False
    waiting_for_text: bool = False


class FeishuStreamingAdapter(builtin.FeishuAdapter):
    draft_stream_is_message = True
    SUPPORTS_MESSAGE_EDITING = True

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.cards = StreamCardController(
            Config(hermes_home(), extra=config.extra),
            FeishuClient(
                FeishuClientConfig(
                    self._app_id,
                    self._app_secret,
                    LARK_DOMAIN if self._domain_name == "lark" else DEFAULT_DOMAIN,
                )
            ),
        )
        self._card_routes: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._processing_sources: ContextVar[tuple[tuple[str, str], ...]] = ContextVar(
            "feishu_card_processing_sources", default=(),
        )
        self._observations: dict[tuple[str, str], CardObservations] = {}
        self._card_loop: asyncio.AbstractEventLoop | None = None
        self._static_cards: OrderedDict[str, None] = OrderedDict()
        ACTIVE.add(self)

    async def handle_message(self, event: MessageEvent) -> None:
        self._card_loop = asyncio.get_running_loop()
        # 引用目标和入站消息身份分开保存，不通过 chat_id 猜测所属用户
        raw = event.raw_message
        body = getattr(getattr(raw, "event", None), "message", None)
        if isinstance(raw, dict):
            body = (raw.get("event") or {}).get("message", raw.get("message", raw))
        real_thread = body.get("thread_id") if isinstance(body, dict) else getattr(body, "thread_id", None)
        if event.reply_to_message_id and event.source.thread_id and body is not None and not real_thread:
            event.source.thread_id = None
        source_id = str(event.message_id or "")
        if source_id:
            self._card_routes[(event.source.chat_id, source_id)] = event.reply_to_message_id or source_id
            while len(self._card_routes) > 1024:
                self._card_routes.popitem(last=False)
        await super().handle_message(event)

    @staticmethod
    def _internal_event(chat_id: str) -> MessageEvent | None:
        # A producer event has its own identity even when its source has no reply anchor.
        bound_event = getattr(session_context, "_SESSION_TURN_EVENT", None)
        event = bound_event.get() if bound_event is not None else None
        if isinstance(event, MessageEvent) and event.internal and event.source.chat_id == chat_id:
            return event
        return None

    def processing_source_id(self, chat_id: str) -> str:
        internal = self._internal_event(chat_id)
        if internal is not None:
            return str(internal.message_id or "")
        stack = self._processing_sources.get()
        if stack:
            source_chat, source_id = stack[-1]
            return source_id if not chat_id or chat_id == source_chat else ""
        source_chat = get_session_env("HERMES_SESSION_CHAT_ID")
        if chat_id and source_chat and chat_id != source_chat:
            return ""
        return str(get_session_env("HERMES_SESSION_MESSAGE_ID") or "")

    def _card_observations(self, chat_id: str, source_id: str, *, create: bool = False) -> CardObservations | None:
        event = self._internal_event(chat_id)
        if event is not None and event.message_id == source_id:
            # The authority executes synthetic turns without processing hooks. The event
            # owns its buffer so silent turns need no adapter cleanup.
            observations = getattr(event, "_cheerwhy_card_observations", None)
            if observations is None:
                observations = CardObservations(waiting_for_text=True)
                vars(event)["_cheerwhy_card_observations"] = observations
            return cast(CardObservations, observations)
        route = (chat_id, source_id)
        return self._observations.setdefault(route, CardObservations()) if create else self._observations.get(route)

    def streaming_overflow_limit(self) -> int | None:
        if (
            self.processing_source_id(get_session_env("HERMES_SESSION_CHAT_ID"))
            and self.cards._cfg.enabled
            and self.cards._cfg.draft_transport
        ):
            # The rich-text limit belongs to the base transport. CardKit owns card
            # splitting and must see the full cumulative text at finalization.
            return maxsize
        return None

    def supports_draft_streaming(
        self, chat_type: str | None = None, metadata: dict[str, Any] | None = None, chat_id: str | None = None,
    ) -> bool:
        chat_id = chat_id or get_session_env("HERMES_SESSION_CHAT_ID")
        source_id = self.processing_source_id(chat_id)
        if not (self.cards._cfg.enabled and source_id and chat_id):
            return False
        try:
            self._card_loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        observations = self._card_observations(chat_id, source_id, create=True)
        assert observations is not None
        observations.streaming = True
        if observations.pending:
            self._card_loop.create_task(self._drain_observations(source_id, chat_id, observations))
        return True

    async def observe_card(self, source_id: str, chat_id: str, kind: str, payload: dict[str, Any]) -> None:
        observations = self._card_observations(chat_id, source_id)
        if observations is None:
            return
        observations.pending.append((kind, payload))
        if observations.streaming:
            await self._drain_observations(source_id, chat_id, observations)

    async def _drain_observations(self, source_id: str, chat_id: str, observations: CardObservations) -> None:
        async with observations.lock:
            while observations.streaming and not observations.waiting_for_text and observations.pending:
                kind, payload = observations.pending.popleft()
                try:
                    await self._apply_observation(source_id, chat_id, kind, payload)
                except Exception:
                    _logger.exception("Could not apply Feishu card observation %s", kind)

    async def _apply_observation(self, source_id: str, chat_id: str, kind: str, payload: dict[str, Any]) -> None:
        if kind in {"start", "reasoning", "tool_start"} and source_id not in self.cards.turns:
            if kind == "reasoning" and not self.cards._cfg.show_reasoning:
                return
            await self._card_frame(source_id, chat_id, "", reply_to=source_id)
        await self.cards.observe(source_id, kind, payload)

    async def _card_frame(
        self, source_id: str, chat_id: str, content: str, *, reply_to: str | None = None,
    ) -> bool:
        internal = self._internal_event(chat_id)
        return await self.cards.frame(
            source_id, chat_id, content, source_id=source_id,
            reply_to=None if internal else self._card_routes.get((chat_id, source_id), reply_to),
            thread_id=internal.source.thread_id if internal else None,
            session_key=get_session_env("HERMES_SESSION_KEY"),
        )

    async def send_draft(
        self, chat_id: str, draft_id: int, content: str, metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        source_id = self.processing_source_id(chat_id)
        observations = self._card_observations(chat_id, source_id)
        if not observations or not observations.streaming:
            return SendResult(success=False, error="Missing Feishu turn identity")
        if content.strip():
            observations.waiting_for_text = False
        if observations.waiting_for_text:
            return SendResult(success=True, raw_response={"skipped": True})
        await self._drain_observations(source_id, chat_id, observations)
        if not observations.streaming:
            return SendResult(success=True, raw_response={"skipped": True})
        ok = await self._card_frame(
            source_id, chat_id, content, reply_to=(metadata or {}).get("reply_to_message_id"),
        )
        message_id = self.cards.message_id(source_id, chat_id)
        if not ok and message_id:
            # Keep an existing card on the draft path if a preview update fails.
            return SendResult(success=True, message_id=message_id, raw_response={"skipped": True})
        return SendResult(success=ok, message_id=message_id)

    async def abandon_open_draft(
        self, chat_id: str, content: str, *, metadata: dict[str, Any] | None = None,
    ) -> None:
        source_id = self.processing_source_id(chat_id)
        observations = self._card_observations(chat_id, source_id)
        self._observations.pop((chat_id, source_id), None)
        if observations:
            observations.streaming = False
            await self.cards.close_source(source_id)

    async def send_final_ledgered(
        self, event: MessageEvent, session_key: str, text_content: str, metadata: dict[str, Any], *,
        reply_to: str | None, is_ephemeral_response: bool = False,
    ) -> tuple[SendResult, BasePlatformAdapter]:
        # The base adapter still owns the durable delivery ledger. Carry the precise
        # inbound ID across the boundary where the agent's context has been reset.
        source_id = event.ledger_message_id or event.message_id
        result = await super().send_final_ledgered(
            event, session_key, text_content, {**metadata, "_card_source_id": source_id},
            reply_to=reply_to, is_ephemeral_response=is_ephemeral_response,
        )
        return cast(tuple[SendResult, BasePlatformAdapter], result)

    async def on_processing_start(self, event: MessageEvent) -> None:
        if event.internal:
            return
        # Queued follow-ups run inside the original gateway context with a new inbound ID.
        if event.message_id:
            route = (event.source.chat_id, event.message_id)
            stack = self._processing_sources.get()
            self._processing_sources.set((*stack, route))
            self._observations.setdefault(route, CardObservations())
        await super().on_processing_start(event)

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        source_id = event.message_id
        if source_id:
            route = (event.source.chat_id, source_id)
            stack = self._processing_sources.get()
            if stack and stack[-1] == route:
                self._processing_sources.set(stack[:-1])
            observations = self._observations.pop(route, None)
            if observations:
                observations.streaming = False
            await self.cards.close_source(
                source_id, aborted=outcome is ProcessingOutcome.CANCELLED, failed=outcome is ProcessingOutcome.FAILURE,
            )
        if not event.internal:
            await super().on_processing_complete(event, outcome)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict | None = None,
    ) -> SendResult:
        md = metadata or {}
        source_id = md.get("_card_source_id") or self.processing_source_id(chat_id)
        if not md.get("non_conversational") and source_id:
            observations = self._card_observations(chat_id, source_id)
            if observations and observations.streaming and (md.get("_interim_send") or md.get("notify")):
                if observations.waiting_for_text and content.strip():
                    observations.waiting_for_text = False
                    await self._card_frame(source_id, chat_id, "")
                await self._drain_observations(source_id, chat_id, observations)
            if md.get("_interim_send"):
                message_id = await self.cards.interim(source_id, content)
                if message_id:
                    return SendResult(success=True, message_id=message_id)
                return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)
            if md.get("notify") and self.cards.message_id(source_id, chat_id):
                if await self.cards.deliver(source_id, chat_id, content):
                    return SendResult(success=True, message_id=self.cards.message_id(source_id, chat_id))
                return SendResult(success=False, error="Could not finalize existing streamed card")
        if md.get("expect_edits"):
            return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)
        # 后台通知按明确目标独立投递，不合并进前台活跃卡片
        if not content.strip():
            return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)
        thread_id = md.get("thread_id")
        anchor = reply_to or md.get("reply_to_message_id")
        topic = {}
        if not anchor and thread_id:
            if str(thread_id).startswith("omt_"):
                topic["thread_id"] = thread_id
            else:
                # Older standalone callers store a reply message ID in this field.
                anchor = thread_id
        try:
            card = build_static_card(content)
            assert self.cards._client is not None
            message_id = await self.cards._client.send_card_to_chat(
                chat_id,
                card,
                reply_to_message_id=anchor,
                **topic,
            )
            self._static_cards[message_id] = None
            while len(self._static_cards) > 512:
                self._static_cards.popitem(last=False)
            return SendResult(success=True, message_id=message_id)
        except Exception:
            _logger.warning("Static card failed; using built-in reply", exc_info=True)
            return await super().send(chat_id, content, reply_to=reply_to, metadata=metadata)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        source_id = self.cards.source_for_message(message_id, chat_id)
        if source_id:
            return await self.send(chat_id, content, metadata={"notify": True, "_card_source_id": source_id})
        if message_id not in self._static_cards:
            return await super().edit_message(chat_id, message_id, content, finalize=finalize)
        try:
            assert self.cards._client is not None
            await self.cards._client.update_card_message(message_id, build_static_card(content))
            return SendResult(success=True, message_id=message_id)
        except Exception:
            _logger.warning("Static card update failed", exc_info=True)
            return SendResult(success=False, error="Card update failed")

    async def send_or_update_status(
        self,
        chat_id: str,
        status_key: str,
        content: str,
        metadata: dict | None = None,
    ) -> SendResult:
        # 状态通报保留明确投递目标，不更新无法可靠关联的卡片
        return await super().send(chat_id, content, metadata=metadata)

    async def interrupt_session_activity(
        self,
        session_key: str,
        chat_id: str,
        metadata: dict | None = None,
    ) -> None:
        sources = {
            turn.source_id
            for turn in self.cards.turns.values()
            if turn.session.session_key == session_key and turn.session.chat_id == chat_id
        }
        for source_id in sources:
            await self.cards.close_source(source_id)

    async def disconnect(self) -> None:
        try:
            for observations in self._observations.values():
                observations.streaming = False
            self._observations.clear()
            await self.cards.close()
        finally:
            ACTIVE.discard(self)
            await super().disconnect()
