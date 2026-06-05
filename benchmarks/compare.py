import argparse
import json
import sys


MATCH_KEYS = ("name", "device", "world_size", "size", "mode", "bucket_cap_mb")


def load_records(path):
    records = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("skipped"):
                continue
            key = tuple(record.get(name) for name in MATCH_KEYS)
            records[key] = record
    return records


def key_name(key):
    return ", ".join(f"{name}={value}" for name, value in zip(MATCH_KEYS, key) if value is not None)


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark JSONL files")
    parser.add_argument("baseline")
    parser.add_argument("current")
    parser.add_argument("--max-regression", type=float, default=1.25)
    parser.add_argument("--metric", default="median_ms")
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()

    baseline = load_records(args.baseline)
    current = load_records(args.current)
    regressions = []

    for key, current_record in sorted(current.items(), key=lambda item: key_name(item[0])):
        baseline_record = baseline.get(key)
        if baseline_record is None:
            print(json.dumps({"status": "new", "key": key_name(key), "current": current_record}, sort_keys=True))
            continue

        baseline_value = baseline_record.get(args.metric)
        current_value = current_record.get(args.metric)
        if baseline_value in (None, 0) or current_value is None:
            continue
        ratio = current_value / baseline_value
        result = {
            "status": "ok" if ratio <= args.max_regression else "regression",
            "key": key_name(key),
            "metric": args.metric,
            "baseline": baseline_value,
            "current": current_value,
            "ratio": ratio,
        }
        print(json.dumps(result, sort_keys=True))
        if ratio > args.max_regression:
            regressions.append(result)

    for key in sorted(set(baseline) - set(current), key=key_name):
        print(json.dumps({"status": "missing", "key": key_name(key)}, sort_keys=True))

    if regressions and args.fail_on_regression:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
