#! python3.7

import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta, timezone
from queue import Queue
from time import sleep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Whisper model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Use multilingual model instead of English-only.")
    parser.add_argument("--energy_threshold", default=1000, type=int,
                        help="Energy level for microphone detection.")
    parser.add_argument("--record_timeout", default=2, type=float,
                        help="Duration for each short recording in seconds.")
    parser.add_argument("--phrase_timeout", default=3, type=float,
                        help="Time of silence before considering a new phrase.")
    parser.add_argument("--list_mics", action="store_true",
                        help="List available microphones and exit.")
    parser.add_argument("--mic", type=int, default=None,
                        help="Microphone device index to use (from list_mics).")
    parser.add_argument("--language", type=str, default=None,
                        help="Force transcription language using ISO 639-1 code (e.g., 'hi' for Hindi). Leave empty for auto-detect.")

    args = parser.parse_args()

    # === Microphone Handling ===
    if args.list_mics:
        print("Available microphone devices:")
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"{index}: {name}")
        return

    # Queue for audio data
    phrase_time = None
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    # Select microphone
    if args.mic is not None:
        source = sr.Microphone(sample_rate=16000, device_index=args.mic)
    else:
        source = sr.Microphone(sample_rate=16000)  # default system mic

    # === Load Whisper Model ===
    model_name = args.model
    if args.model != "large" and not args.non_english:
        model_name += ".en"  # English-only
    audio_model = whisper.load_model(model_name)

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout
    transcription = [""]

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        """Receive audio and store in queue."""
        data = audio.get_raw_data()
        data_queue.put(data)

    # Start background recording
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded. Listening... (Press Ctrl+C to stop)\n")

    # === Main Loop ===
    while True:
        try:
            now = datetime.now(timezone.utc)

            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now

                audio_data = b"".join(data_queue.queue)
                data_queue.queue.clear()

                # Convert to float32 numpy array
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Transcribe with Whisper
                kwargs = {"fp16": torch.cuda.is_available()}
                if args.language:
                    kwargs["language"] = args.language  # Force language
                result = audio_model.transcribe(audio_np, **kwargs)
                text = result["text"].strip()

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                # Refresh screen with latest transcription
                os.system("cls" if os.name == "nt" else "clear")
                for line in transcription:
                    print(line)
                print("", end="", flush=True)

            else:
                sleep(0.25)

        except KeyboardInterrupt:
            break

    print("\n\nFinal Transcription:")
    for line in transcription:
        print(line)


if __name__ == "__main__":
    main()
