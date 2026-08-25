import unittest
from wholebody.graph.node import Node, Edge, Port
from wholebody.graph.model_graph import ModelGraph
from wholebody.graph.exporter import to_json, export_dot


class TestModelGraph(unittest.TestCase):

    def test_graph_validation_and_export(self):
        graph = ModelGraph("PoseDemoGraph")

        in_node = Node(id="input", category="input", registry_name="InputTensor")
        bb_node = Node(id="backbone", category="backbone", registry_name="SimpleCNN", params={"in_channels": 3, "stage_channels": [32, 64]})
        hd_node = Node(id="head", category="head", registry_name="HeatmapHead", params={"in_channels": 64, "num_keypoints": 133})

        graph.add_node(in_node)
        graph.add_node(bb_node)
        graph.add_node(hd_node)

        graph.add_edge(Edge(source_node="input", source_port="out", target_node="backbone", target_port="in"))
        graph.add_edge(Edge(source_node="backbone", source_port="out", target_node="head", target_port="in"))

        order = graph.validate_topology()
        self.assertEqual(order, ["input", "backbone", "head"])

        cfg_dict = graph.to_config()
        self.assertEqual(cfg_dict["model"]["backbone"]["type"], "SimpleCNN")
        self.assertEqual(cfg_dict["model"]["head"]["num_keypoints"], 133)

        # JSON export test
        json_data = to_json(graph)
        self.assertEqual(len(json_data["nodes"]), 3)
        self.assertEqual(len(json_data["edges"]), 2)

        # DOT export test
        dot_str = export_dot(graph)
        self.assertIn("SimpleCNN", dot_str)


if __name__ == "__main__":
    unittest.main()
