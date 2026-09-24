from gymnasium import Env

from gym_multigrid.core.agent import DefaultActions


class ManualControl:
    def __init__(
        self,
        env: Env,
        seed: int | None = None,
    ) -> None:
        self.env: Env = env

        self.seed: int | None = seed

        self.closed: bool = False

    def start(self) -> None:
        self.reset(self.seed)

        while not self.closed:
            key = input()

            self.key_handler(key)

    def reset(self, seed: int | None = None) -> None:
        self.env.reset(seed=seed)

        self.env.render()

    def step(self, actions: list[int]) -> None:
        _, _, terminated, truncated, _ = self.env.step(actions)

        if terminated:
            print("Terminated")

            self.reset(self.seed)

        elif truncated:
            print("truncated")

            self.reset(self.seed)

        else:
            self.env.render()

    def key_handler(self, event: str) -> None:
        key: str = event

        print("pressed", key)

        if key == "escape":
            self.env.close()

            return

        if key == "backspace":
            self.reset()

            return

        key_to_action: dict[str, int] = {
            "left": DefaultActions.LEFT,
            "right": DefaultActions.RIGHT,
            "up": DefaultActions.FORWARD,
            "space": DefaultActions.TOGGLE,
            "pageup": DefaultActions.PICKUP,
            "pagedown": DefaultActions.DROP,
            "tab": DefaultActions.PICKUP,
            "left shift": DefaultActions.DROP,
            "enter": DefaultActions.DONE,
        }

        if key in key_to_action.keys():
            action = [key_to_action[key]]

            self.step(action)

        else:
            print(key)
