"""Line protocol inside the isolated predictor container.

Only feature dictionaries and training labels arrive over stdin. The held-out
source file and future labels never enter this container.
"""

import contextlib
import importlib.util
import json
import sys


def _reply(value: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    spec = importlib.util.spec_from_file_location("isolated_predictor", "/workspace/predictor.py")
    if spec is None or spec.loader is None:
        return 2
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
        predictor = module.Predictor()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request["command"]
            if command == "stop":
                _reply({"ok": True})
                return 0
            with contextlib.redirect_stdout(sys.stderr):
                if command == "fit":
                    predictor.fit(request["feature"], request["outcome"])
                    answer: dict[str, object] = {"ok": True}
                elif command == "predict":
                    answer = {"probability": predictor.predict(request["feature"])}
                else:
                    raise ValueError("invalid command")
            _reply(answer)
        except Exception as exc:
            _reply({"error": type(exc).__name__})
            return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
