"""Example: receive RoboVoice output in another Python file."""
from queue import Queue, Empty
from threading import Event, Thread

from RoboVoice_Main import run_gui
from voice_engine import SAMPLERATE


def consume(blocks, stopped):
    while not stopped.is_set():
        try:
            samples = blocks.get(timeout=0.2)
        except Empty:
            continue
        # Replace this with your own Python audio handling. Each array contains
        # mono float32 samples at SAMPLERATE Hz, nominally in the range [-1, 1].
        # File writes, network sends, and other slow work belong here, not in
        # the sounddevice callback. This example intentionally discards samples.
        handle_audio(samples, SAMPLERATE)


def handle_audio(samples, sample_rate):
    pass


if __name__ == '__main__':
    blocks = Queue(maxsize=8)
    stopped = Event()
    worker = Thread(target=consume, args=(blocks, stopped), daemon=True)
    worker.start()
    try:
        # Tk must run on the main thread. Set monitor=False to silence speakers.
        run_gui(output_queue=blocks, monitor=True)
    finally:
        stopped.set()
        worker.join(timeout=1)
