import argparse
import json
import sys
from pathlib import Path

# Ensure project source directory is on the import path
proj_root = Path(__file__).resolve().parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from source import db_operations as db_ops
from source.common_utils import generate_unique_task_id as sm_generate_unique_task_id


def _load_params(param_arg: str) -> dict:
    """Load params from a JSON string or from a @file reference."""
    if param_arg.startswith("@"):
        file_path = Path(param_arg[1:]).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"Params file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    # Otherwise, treat as literal JSON string
    return json.loads(param_arg)


def main() -> None:
    parser = argparse.ArgumentParser(
        "add_task – enqueue a task for the Wan2GP headless server"
    )
    parser.add_argument(
        "--type",
        required=True,
        help="Task type string (e.g. travel_orchestrator, travel_segment, generate_openpose, …)",
    )
    parser.add_argument(
        "--params",
        required=True,
        help="JSON string with task payload OR @<path-to-json-file>",
    )
    parser.add_argument(
        "--dependant-on",
        dest="dependant_on",
        default=None,
        help="Optional task_id that this new task depends on.",
    )
    args = parser.parse_args()

    try:
        payload_dict = _load_params(args.params)
    except Exception as e:
        print(f"[ERROR] Could not parse --params: {e}")
        sys.exit(1)

    # Auto-generate task_id if needed
    if "task_id" not in payload_dict or not payload_dict["task_id"]:
        payload_dict["task_id"] = sm_generate_unique_task_id(f"{args.type[:8]}_")
        print(f"[INFO] task_id not supplied – generated: {payload_dict['task_id']}")

    try:
        db_ops.add_task_to_db(
            task_payload=payload_dict,
            task_type_str=args.type,
            dependant_on=args.dependant_on,
        )
        print(f"[SUCCESS] Task {payload_dict['task_id']} ({args.type}) enqueued.")
    except Exception as e_db:
        print(f"[ERROR] Failed to enqueue task: {e_db}")
        sys.exit(2)


if __name__ == "__main__":
    main()
