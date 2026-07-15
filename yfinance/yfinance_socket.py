import yfinance as yf
import uuid
import json
import os

OUTPUT_DIR = "/data/yfinance"
SYMBOLS = ["VXUS", "VUG", "VTI", "SOXX", "FMAT", "IVV"]
BUFFER_LIMIT = 10

buffer = []

def flush_buffer():
    global buffer

    filename = f"{OUTPUT_DIR}/batch_{uuid.uuid4().hex}.json"
    tmp_filename = filename + ".tmp"
    with open(tmp_filename, "w") as f:
        for message in buffer:
            f.write(json.dumps(message) + "\n")
    os.rename(tmp_filename, filename)
    buffer = []


def message_handler(message):
    print(message)
    global buffer
    buffer.append(message)
    
    if len(buffer) > BUFFER_LIMIT:
        flush_buffer()


def main():
    ws = yf.WebSocket()
    try:
        ws.subscribe(SYMBOLS)
        ws.listen(message_handler)
    finally:
        flush_buffer()


if __name__ == "__main__":
    main()