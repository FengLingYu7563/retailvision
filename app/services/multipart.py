"""串流 multipart 解析：邊讀邊累計每個檔案的大小（驗收 2.3，Codex 第 3 輪 L2）。

為什麼不用 FastAPI 的 UploadFile：FastAPI 在呼叫 endpoint 之前，會把整個 multipart 解析完、
每個檔案都寫進暫存檔，之後才輪得到我們檢查大小。批次的總量上限是 100 MB，
一個 90 MB 的單檔在被拒絕前就已經完整寫進硬碟。

這裡直接讀 request.stream()，用 python-multipart 的串流解析器處理：
- 每個檔案超過 max_part_bytes 時，標記為過大、**丟掉後面的資料**（不放記憶體、不寫硬碟），
  由呼叫端決定是整個請求 413（單張上傳）還是記成該項目的錯誤（批次）。
- 檔案數超過上限、一般欄位過大、欄位數過多：直接回錯誤，停止讀取。
整個請求的總量另外由 app/core/bodylimit.py 在更外層限制。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fastapi import Request
from python_multipart.multipart import MultipartParser, parse_options_header

from app.core.errors import AppError

MAX_FIELD_BYTES = 64 * 1024
MAX_FIELDS = 20


@dataclass
class FilePart:
    name: str
    filename: str | None
    data: bytearray = field(default_factory=bytearray)
    too_large: bool = False


@dataclass
class ParsedForm:
    fields: dict[str, str]
    files: list[FilePart]

    def files_named(self, name: str) -> list[FilePart]:
        return [f for f in self.files if f.name == name]

    def field(self, name: str) -> str | None:
        return self.fields.get(name)


async def read_multipart(request: Request, max_part_bytes: int, max_files: int) -> ParsedForm:
    ctype, params = parse_options_header(request.headers.get("content-type", ""))
    if ctype != b"multipart/form-data" or b"boundary" not in params:
        raise AppError("VALIDATION_ERROR", "請求必須是 multipart/form-data", 422)

    fields: dict[str, str] = {}
    files: list[FilePart] = []
    state: dict[str, object] = {}
    headers: dict[bytes, bytes] = {}
    header_field = bytearray()
    header_value = bytearray()

    def on_part_begin() -> None:
        headers.clear()
        state.clear()

    def on_header_field(data: bytes, start: int, end: int) -> None:
        header_field.extend(data[start:end])

    def on_header_value(data: bytes, start: int, end: int) -> None:
        header_value.extend(data[start:end])

    def on_header_end() -> None:
        headers[bytes(header_field).lower()] = bytes(header_value)
        header_field.clear()
        header_value.clear()

    def on_headers_finished() -> None:
        _, disp = parse_options_header(headers.get(b"content-disposition", b""))
        name = disp.get(b"name", b"").decode("utf-8", "replace")
        if b"filename" in disp:
            if len(files) >= max_files:
                raise AppError("VALIDATION_ERROR", f"一次最多 {max_files} 個檔案", 422)
            part = FilePart(name, disp[b"filename"].decode("utf-8", "replace"))
            files.append(part)
            state["file"] = part
        else:
            if len(fields) >= MAX_FIELDS:
                raise AppError("VALIDATION_ERROR", "表單欄位過多", 422)
            state["field"] = name
            state["value"] = bytearray()

    def on_part_data(data: bytes, start: int, end: int) -> None:
        part = state.get("file")
        if isinstance(part, FilePart):
            if part.too_large:
                return  # 已經超過上限：後面的資料直接丟掉
            if len(part.data) + (end - start) > max_part_bytes:
                part.too_large = True
                part.data = bytearray()  # 釋放已經讀到的部分
                return
            part.data.extend(data[start:end])
        else:
            value = state["value"]
            assert isinstance(value, bytearray)
            if len(value) + (end - start) > MAX_FIELD_BYTES:
                raise AppError("FILE_TOO_LARGE", "表單欄位過大", 413)
            value.extend(data[start:end])

    def on_part_end() -> None:
        if "field" in state:
            value = state["value"]
            assert isinstance(value, bytearray)
            fields[str(state["field"])] = value.decode("utf-8", "replace")

    parser = MultipartParser(
        params[b"boundary"],
        {
            "on_part_begin": on_part_begin,
            "on_part_data": on_part_data,
            "on_part_end": on_part_end,
            "on_header_field": on_header_field,
            "on_header_value": on_header_value,
            "on_header_end": on_header_end,
            "on_headers_finished": on_headers_finished,
        },
    )
    try:
        async for chunk in request.stream():
            parser.write(chunk)
        parser.finalize()
    except AppError:
        raise
    except Exception as exc:  # python-multipart 對格式錯誤的請求會丟出各種例外
        raise AppError("VALIDATION_ERROR", "multipart 格式錯誤", 422) from exc
    return ParsedForm(fields, files)
