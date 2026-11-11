class NotifyAwareErr(Exception):
    def __init__(self, message: str, notify: bool = True) -> None:
        super().__init__(message)
        self.notify: bool = notify


class FatalErr(NotifyAwareErr):
    pass


class RetriableErr(Exception):
    pass


class ExitableErr(NotifyAwareErr, RetriableErr):
    def __init__(self, message: str, exit_code: int = 1, notify: bool = True) -> None:
        super().__init__(message, notify=notify)
        self.exit_code: int = exit_code


class PayloadErr(RetriableErr):
    def __init__(self, message: str, payload: list[int | str]) -> None:
        super().__init__(message)
        self.payload: list[int | str] = payload


class CmdErr(RetriableErr):
    def __init__(self, message: str, code: int | None, stderr: str) -> None:
        super().__init__(message)
        self.code: int | None = code
        self.stderr: str = stderr

