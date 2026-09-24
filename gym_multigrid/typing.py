from typing import Generic, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, SerializationInfo, model_serializer
from pyparsing import Any

Position: TypeAlias = tuple[int, int]
Size: TypeAlias = tuple[int, int]

EnvKwargsType = TypeVar("EnvKwargsType", bound=BaseModel)


class EnvMakeConfig(BaseModel, Generic[EnvKwargsType]):
    id: str = "multigrid-rooms-v0"
    max_episode_steps: int | None
    disable_env_checker: bool | None = None
    env_kwargs: EnvKwargsType

    # allow arbitrary kwargs
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_serializer
    def serialize(self, info: SerializationInfo) -> dict[str, Any]:
        """Serialize the model to a dictionary."""
        context = info.context
        if context:
            if context.get("flatten", False):
                return {
                    "id": self.id,
                    "max_episode_steps": self.max_episode_steps,
                    "disable_env_checker": self.disable_env_checker,
                    **self.env_kwargs.model_dump(),
                }

        return {
            "id": self.id,
            "max_episode_steps": self.max_episode_steps,
            "disable_env_checker": self.disable_env_checker,
            "env_kwargs": self.env_kwargs.model_dump(),
        }
