"""对话记忆后端的统一契约。"""

from abc import ABC, abstractmethod

from customer_care.state import CareState


class VersionNotFoundError(LookupError):
    """请求的对话快照不存在时抛出。"""


class MemoryStore(ABC):
    @abstractmethod
    async def load(self, user_id: str) -> CareState:
        """读取当前状态；新用户返回空状态。"""

    @abstractmethod
    async def save(self, user_id: str, state: CareState) -> str:
        """保存不可变的新快照，并返回版本标识。"""

    @abstractmethod
    async def history(self, user_id: str) -> list[dict[str, object]]:
        """返回当前状态中的消息列表。"""

    @abstractmethod
    async def rollback(self, user_id: str, version: str) -> CareState:
        """将已保存快照设为当前状态，但不删除后续快照。"""

    @abstractmethod
    async def claim_confirmation(
        self,
        user_id: str,
        confirmation_id: str,
        token_hash: str,
        now: float,
        target_status: str,
    ) -> tuple[str, CareState]:
        """原子校验并消费待确认操作，返回结果码及最新状态。"""

