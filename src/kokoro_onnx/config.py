import json
from pathlib import Path

MAX_PHONEME_LENGTH = 100
SAMPLE_RATE = 24000


class KoKoroConfig:
    def __init__(
        self,
        model_path: str,
        voice_path: str,
    ):
        self.model_path = model_path
        self.voice_path = voice_path

    def validate(self):
        if not Path(self.voice_path).exists():
            error_msg = f"Voices file not found at {self.voice_path}"
            raise FileNotFoundError(error_msg)

        if not Path(self.model_path).exists():
            error_msg = f"Model file not found at {self.model_path}"
            error_msg += "\nYou can download the model file from https://github.com/thewh1teagle/kokoro-onnx/releases"
            raise FileNotFoundError(error_msg)


def get_vocab():
    with open(Path(__file__).parent / "config.json", encoding="utf-8") as fp:
        config = json.load(fp)
        return config["vocab"]


DEFAULT_VOCAB = get_vocab()
