"""parquet_lite — 새 패키지 없이 평평한 parquet 파일을 읽는 작은 읽개.

받은 원자료(VitaminC · MiniCheck)가 parquet 이고 이 PC 에는 pyarrow 가 없다.
패키지 추가는 승인 몫이라, 필요한 만큼만 직접 읽는다.

되는 것 : 평평한 스키마(중첩 없음) · 압축 없음/snappy · 데이터 페이지 v1·v2 ·
PLAIN · 사전(RLE_DICTIONARY · PLAIN_DICTIONARY) 인코딩 · BYTE_ARRAY · INT32 · INT64 · BOOLEAN.
안 되는 것은 ValueError 로 멈춘다 (조용히 틀린 값을 내지 않는다).
"""
import struct

# thrift compact 꼴 번호
_T_TRUE, _T_FALSE, _T_BYTE, _T_I16, _T_I32, _T_I64 = 1, 2, 3, 4, 5, 6
_T_DOUBLE, _T_BINARY, _T_LIST, _T_SET, _T_MAP, _T_STRUCT = 7, 8, 9, 10, 11, 12

# parquet 값 꼴 · 인코딩 · 압축 · 페이지 번호
_BOOLEAN, _INT32, _INT64, _BYTE_ARRAY = 0, 1, 2, 6
_PLAIN, _PLAIN_DICTIONARY, _RLE_DICTIONARY = 0, 2, 8
_UNCOMPRESSED, _SNAPPY = 0, 1
_DATA_PAGE, _DICTIONARY_PAGE, _DATA_PAGE_V2 = 0, 2, 3
_OPTIONAL = 1


def _varint(buf, pos):
    shift = 0
    value = 0
    while True:
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if b < 0x80:
            return value, pos
        shift += 7


def _zigzag(n):
    return (n >> 1) ^ -(n & 1)


class _Thrift:
    """thrift compact 구조체를 {필드 번호: 값} 사전으로 푼다."""

    def __init__(self, buf, pos=0):
        self.buf = buf
        self.pos = pos

    def read_struct(self):
        out = {}
        last = 0
        while True:
            head = self.buf[self.pos]
            self.pos += 1
            if head == 0:
                return out
            kind = head & 0x0F
            delta = head >> 4
            if delta == 0:
                raw, self.pos = _varint(self.buf, self.pos)
                field = _zigzag(raw)
            else:
                field = last + delta
            last = field
            out[field] = self._value(kind)

    def _value(self, kind):
        if kind == _T_TRUE:
            return True
        if kind == _T_FALSE:
            return False
        if kind == _T_BYTE:
            self.pos += 1
            return self.buf[self.pos - 1]
        if kind in (_T_I16, _T_I32, _T_I64):
            raw, self.pos = _varint(self.buf, self.pos)
            return _zigzag(raw)
        if kind == _T_DOUBLE:
            self.pos += 8
            return struct.unpack("<d", self.buf[self.pos - 8:self.pos])[0]
        if kind == _T_BINARY:
            size, self.pos = _varint(self.buf, self.pos)
            self.pos += size
            return bytes(self.buf[self.pos - size:self.pos])
        if kind in (_T_LIST, _T_SET):
            return self._list()
        if kind == _T_MAP:
            return self._map()
        if kind == _T_STRUCT:
            return self.read_struct()
        raise ValueError(f"thrift 꼴 {kind} 은 모른다")

    def _list(self):
        head = self.buf[self.pos]
        self.pos += 1
        size = head >> 4
        kind = head & 0x0F
        if size == 15:
            size, self.pos = _varint(self.buf, self.pos)
        items = []
        for _ in range(size):
            if kind in (_T_TRUE, _T_FALSE):
                items.append(self.buf[self.pos] == 1)
                self.pos += 1
            else:
                items.append(self._value(kind))
        return items

    def _map(self):
        size, self.pos = _varint(self.buf, self.pos)
        if size == 0:
            return {}
        kinds = self.buf[self.pos]
        self.pos += 1
        out = {}
        for _ in range(size):
            key = self._value(kinds >> 4)
            out[key] = self._value(kinds & 0x0F)
        return out


def snappy_decompress(buf):
    """snappy 원형(틀 없는) 압축을 푼다."""
    size, pos = _varint(buf, 0)
    out = bytearray()
    n = len(buf)
    while pos < n:
        tag = buf[pos]
        pos += 1
        kind = tag & 3
        if kind == 0:
            length = tag >> 2
            if length >= 60:
                extra = length - 59
                length = int.from_bytes(buf[pos:pos + extra], "little")
                pos += extra
            length += 1
            out += buf[pos:pos + length]
            pos += length
            continue
        if kind == 1:
            length = ((tag >> 2) & 7) + 4
            offset = ((tag >> 5) << 8) | buf[pos]
            pos += 1
        elif kind == 2:
            length = (tag >> 2) + 1
            offset = int.from_bytes(buf[pos:pos + 2], "little")
            pos += 2
        else:
            length = (tag >> 2) + 1
            offset = int.from_bytes(buf[pos:pos + 4], "little")
            pos += 4
        if offset == 0 or offset > len(out):
            raise ValueError("snappy 되짚기 거리가 틀렸다")
        start = len(out) - offset
        if offset >= length:
            out += out[start:start + length]
        else:
            for i in range(length):
                out.append(out[start + i])
    if len(out) != size:
        raise ValueError(f"snappy 푼 길이 {len(out)} ≠ 머리 {size}")
    return bytes(out)


def rle_hybrid(buf, pos, end, bit_width, count):
    """RLE · 비트 묶음 섞임 인코딩에서 값 count 개를 읽는다."""
    values = []
    byte_width = (bit_width + 7) // 8
    mask = (1 << bit_width) - 1
    while len(values) < count and pos < end:
        header, pos = _varint(buf, pos)
        if header & 1:
            groups = header >> 1
            nbytes = groups * bit_width
            chunk = int.from_bytes(buf[pos:pos + nbytes], "little")
            pos += nbytes
            for i in range(groups * 8):
                values.append((chunk >> (i * bit_width)) & mask)
        else:
            run = header >> 1
            value = int.from_bytes(buf[pos:pos + byte_width], "little") if byte_width else 0
            pos += byte_width
            values.extend([value] * run)
    return values[:count]


def _plain(buf, kind, count):
    values = []
    pos = 0
    if kind == _BYTE_ARRAY:
        for _ in range(count):
            size = struct.unpack_from("<I", buf, pos)[0]
            pos += 4
            values.append(buf[pos:pos + size].decode("utf-8"))
            pos += size
        return values
    if kind == _INT64:
        return list(struct.unpack_from(f"<{count}q", buf, 0))
    if kind == _INT32:
        return list(struct.unpack_from(f"<{count}i", buf, 0))
    if kind == _BOOLEAN:
        return [bool((buf[i // 8] >> (i % 8)) & 1) for i in range(count)]
    raise ValueError(f"parquet 값 꼴 {kind} 은 안 읽는다")


def _decompress(codec, body, size):
    if codec == _UNCOMPRESSED:
        return body
    if codec == _SNAPPY:
        out = snappy_decompress(body)
        if len(out) != size:
            raise ValueError("페이지 푼 길이가 머리와 다르다")
        return out
    raise ValueError(f"압축 {codec} 은 안 푼다")


def _read_column(data, meta, kind, optional):
    codec = meta[4]
    total = meta[5]
    pos = meta.get(11) or meta[9]
    dictionary = None
    values = []
    while len(values) < total:
        reader = _Thrift(data, pos)
        header = reader.read_struct()
        pos = reader.pos
        page_type, raw_size, packed_size = header[1], header[2], header[3]
        body = data[pos:pos + packed_size]
        pos += packed_size
        if page_type == _DICTIONARY_PAGE:
            page = _decompress(codec, body, raw_size)
            dictionary = _plain(page, kind, header[7][1])
            continue
        if page_type == _DATA_PAGE:
            page = _decompress(codec, body, raw_size)
            info = header[5]
            num, encoding = info[1], info[2]
            cursor = 0
            levels = [1] * num
            if optional:
                level_len = struct.unpack_from("<I", page, 0)[0]
                levels = rle_hybrid(page, 4, 4 + level_len, 1, num)
                cursor = 4 + level_len
            payload = page[cursor:]
        elif page_type == _DATA_PAGE_V2:
            info = header[8]
            num, encoding = info[1], info[4]
            def_len, rep_len = info[5], info[6]
            if rep_len:
                raise ValueError("반복 칸(중첩)은 안 읽는다")
            levels = rle_hybrid(body, 0, def_len, 1, num) if optional else [1] * num
            rest = body[def_len:]
            compressed = info.get(7, True)
            payload = _decompress(codec, rest, raw_size - def_len) if compressed else rest
        else:
            raise ValueError(f"페이지 꼴 {page_type} 은 안 읽는다")
        present = sum(levels)
        if encoding == _PLAIN:
            got = _plain(payload, kind, present)
        elif encoding in (_RLE_DICTIONARY, _PLAIN_DICTIONARY):
            if dictionary is None:
                raise ValueError("사전 페이지 없이 사전 인코딩이 왔다")
            indices = rle_hybrid(payload, 1, len(payload), payload[0], present)
            got = [dictionary[i] for i in indices]
        else:
            raise ValueError(f"인코딩 {encoding} 은 안 읽는다")
        it = iter(got)
        values.extend(next(it) if level else None for level in levels)
    return values


def read_rows(path):
    """parquet 파일을 칸 이름 → 값 사전의 목록으로 읽는다."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"PAR1" or data[-4:] != b"PAR1":
        raise ValueError(f"parquet 아님 : {path}")
    footer_len = struct.unpack("<I", data[-8:-4])[0]
    meta = _Thrift(data, len(data) - 8 - footer_len).read_struct()
    schema = meta[2]
    if any(el.get(5) for el in schema[1:]):
        raise ValueError("중첩 스키마는 안 읽는다")
    columns = {el[4].decode("utf-8"): el for el in schema[1:]}
    rows = []
    for group in meta[4]:
        table = {}
        for chunk in group[1]:
            cmeta = chunk[3]
            name = cmeta[3][0].decode("utf-8")
            element = columns[name]
            optional = element.get(3) == _OPTIONAL
            table[name] = _read_column(data, cmeta, element[1], optional)
        names = list(table)
        for i in range(group[3]):
            rows.append({name: table[name][i] for name in names})
    return rows
