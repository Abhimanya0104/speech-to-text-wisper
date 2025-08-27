

# Real-time Speech-to-Text with Whisper (Local)

This project provides a **real-time speech-to-text transcription** tool using OpenAI’s **Whisper** model locally, without requiring any API keys. Audio is captured from your microphone and transcribed continuously in the terminal.

---

## **Features**

* Real-time transcription from your microphone.
* Multi-line transcription with automatic phrase detection.
* Adjustable recording and phrase timeout settings.
* Supports multiple Whisper models (`tiny`, `base`, `small`, `medium`, `large`).
* Works fully **offline**; no API key required.
* Cross-platform: Windows, Linux, macOS.

---

## **Requirements**

* Python 3.7+
* Packages:

```bash
pip install numpy torch whisper SpeechRecognition
```

* Microphone connected to your computer.

> For GPU acceleration with Whisper, ensure **PyTorch with CUDA** is installed.

---

## **Usage**

### 1. List available microphones

```bash
python transcribe_demo.py --list_mics
```

This will show all microphones and their indices. Choose the one you want to use.

---

### 2. Run the transcription

```bash
python transcribe_demo.py --mic <MIC_INDEX>
```

Example:

```bash
python transcribe_demo.py --mic 1 --model small --energy_threshold 1000 --record_timeout 2 --phrase_timeout 3
```

**Parameters:**

| Argument             | Description                                                       |
| -------------------- | ----------------------------------------------------------------- |
| `--model`            | Whisper model to use (`tiny`, `base`, `small`, `medium`, `large`) |
| `--non_english`      | Use a multilingual model instead of English-only                  |
| `--energy_threshold` | Microphone energy threshold for detecting speech                  |
| `--record_timeout`   | Maximum seconds to record in one segment                          |
| `--phrase_timeout`   | Seconds of silence to consider a new line in transcription        |
| `--mic`              | Microphone index from `--list_mics` output                        |

---

### 3. Control

* Press **Ctrl+C** to stop recording.
* The terminal will show **final transcription** at the end.

---

## **How It Works**

1. **Microphone Setup:** Uses `SpeechRecognition` to capture audio in real-time.
2. **Queue System:** Audio chunks are stored in a thread-safe queue.
3. **Whisper Transcription:** Audio is converted to `float32` and sent to Whisper.
4. **Phrase Detection:** If a pause is detected, a new line is created in the transcription.
5. **Real-time Display:** Transcription is continuously printed in the terminal.

---

## **Example Output**

```
Model loaded. Listening... (Press Ctrl+C to stop)

Hello everyone
Welcome to my interview demo
We are testing real-time speech transcription
```

---

## **Notes**

* For best results, use a **good microphone** and **quiet environment**.
* Use smaller Whisper models (`tiny`, `base`) for faster processing on CPU.
* For GPU, ensure PyTorch detects CUDA for faster transcription (`fp16=True` is enabled automatically).

