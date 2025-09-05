import time
import asyncio

import sounddevice as sd

from kokoro_onnx import Kokoro


async def main_streaming(model, text):
    t0 = time.time()
    async for audio, sample_rate in model.create_stream(text):
        print("Playing", f"({time.time() - t0:.4f}s to generate)")
        sd.play(audio, sample_rate)
        sd.wait()
        t0 = time.time()


def main_blocking(model, text):
    t0 = time.time()
    audio, sample_rate = model.create(text)
    print("Playing", f"({time.time() - t0:.4f}s to generate)")
    sd.play(audio, sample_rate)
    sd.wait()


if __name__ == "__main__":
    text = """金融機関や大企業などが集中し、新聞・放送・出版などの文化面、\
大学・研究機関などの教育・学術面においても日本の中枢をなす。交通面でも鉄道網 \
道路網、航空路の中心である。 """

    model = Kokoro(
        model_path="model/kokoro-v1.0.int8.onnx", 
        voice_path="model/jf_alpha.bin"
    )

    #
    # Example: Two options for TTS generation.
    #

    # 1. Streaming: play TTS chunks while asynchronously generating next chunk
    print("1. Streaming generation")
    asyncio.run(main_streaming(model, text))

    # 2. Blocking: block until the entire TTS audio is generated
    print("2. Synchronous generation")
    main_blocking(model, text)
