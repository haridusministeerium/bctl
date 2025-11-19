class FatalErr(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)


class RetriableException(Exception):
    pass


class ExitableErr(RetriableException):
    def __init__(self, message, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code: int = exit_code


class PayloadErr(RetriableException):
    def __init__(self, message, payload: list) -> None:
        super().__init__(message)
        self.payload = payload


class CmdErr(RetriableException):
    def __init__(self, message, code: int | None, stderr: str) -> None:
        super().__init__(message)
        self.code = code
        self.stderr = stderr
