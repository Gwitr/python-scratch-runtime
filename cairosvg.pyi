from typing import IO

def svg2png(url: str | None = None, file_obj: IO[bytes] | None = None, bytestring: bytes | bytearray | None = None,
            parent_width: int | float | None = None, parent_height: int | float | None = None, dpi: int | float | None = None,
            scale: int | float | None = None, unsafe: bool | None = None, output_width: int | float | None = None,
            output_height: int | float | None = None, write_to: str | IO[bytes] | None = None) -> bytes | None: ...
