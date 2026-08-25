import argparse
from pathlib import Path
import json

from wholebody.core.config import Config
from wholebody.graph.model_graph import ModelGraph
from wholebody.graph.exporter import export_dot, to_json
from wholebody.utils.logger import get_logger

logger = get_logger("wholebody.tools.export_graph")


def parse_args():
    parser = argparse.ArgumentParser(description="Export Model Architecture to Graph IR (JSON / DOT for GUI)")
    parser.add_argument("--config", type=str, required=True, help="Path to model config YAML")
    parser.add_argument("--output", type=str, default="./work_dirs/model_graph.json", help="Path to output JSON/DOT")
    parser.add_argument("--format", type=str, default="json", choices=["json", "dot"], help="Export format")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = Config.from_file(args.config)
    graph = ModelGraph.from_config(cfg)
    order = graph.validate_topology()
    logger.info(f"Successfully constructed ModelGraph IR. Topological execution order: {order}")

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        json_data = to_json(graph, filepath=out_p)
        logger.info(f"Saved React Flow / Web GUI schema to: {out_p}")
    else:
        dot_str = export_dot(graph, filepath=out_p)
        logger.info(f"Saved Graphviz DOT file to: {out_p}")


if __name__ == "__main__":
    main()
