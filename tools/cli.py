import sys
import subprocess


def main():
    """Unified CLI dispatcher for the WholeBody framework."""
    if len(sys.argv) < 2:
        print("WholeBody Framework CLI")
        print("Usage: wholebody <command> [options]")
        print("\nCommands:")
        print("  train         Train a pose estimation model")
        print("  infer         Run inference on image/video")
        print("  export-graph  Export model graph IR (JSON/DOT for GUI)")
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    cmd_map = {
        "train": "tools/train.py",
        "infer": "tools/infer.py",
        "export-graph": "tools/export_graph.py",
    }

    if command not in cmd_map:
        print(f"Unknown command: '{command}'. Available: {list(cmd_map.keys())}")
        sys.exit(1)

    script_path = cmd_map[command]
    full_cmd = [sys.executable, script_path] + args
    res = subprocess.run(full_cmd)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
