import orjson


def save_to_json(data: dict, output_file: str):
    with open(output_file, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY))


def load_from_json(file):
    with open(file, "rb") as f:
        return orjson.loads(f.read())
