from __future__ import annotations

import unittest

from afts_arc.blind import BlindTask
from afts_arc.grid import as_grid
from afts_arc.hybrid import (
    CanvasNode,
    CorrespondObjectsNode,
    ObjectOperationNode,
    ParseObjectsNode,
    RenderObjectsNode,
    ScenePipelineProgram,
    SelectObjectsNode,
    evaluate_hypothesis,
    execute_object_code_program,
    extract_scene_graph,
    make_object_code_hypothesis,
    object_code_program_from_json,
    object_correspondences,
    synthesize_object_code_programs,
    typed_repair_frontier,
)
from afts_arc.task import ARCPair


def _blind(
    source: list[list[int]],
    target: list[list[int]],
    query: list[list[int]] | None = None,
) -> BlindTask:
    return BlindTask.from_observations(
        train=(ARCPair(as_grid(source), as_grid(target)),),
        test_inputs=(as_grid(source if query is None else query),),
    )


def _crop_program(
    *,
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    role: str = "smallest_area",
) -> ScenePipelineProgram:
    return ScenePipelineProgram(
        ParseObjectsNode(0, 4, "color_groups"),
        CorrespondObjectsNode(),
        SelectObjectsNode(role),
        ObjectOperationNode("crop"),
        CanvasNode("bbox", 0, padding),
        RenderObjectsNode("source_crop"),
    )


class SceneGraphTests(unittest.TestCase):
    def test_scene_graph_has_deterministic_component_and_color_group_roles(self) -> None:
        grid = as_grid(
            [
                [0, 1, 0, 1, 0],
                [0, 1, 0, 0, 0],
                [2, 0, 2, 2, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        components = extract_scene_graph(
            grid,
            background=0,
            connectivity=4,
            grouping="monochrome_components",
        )
        color_groups = extract_scene_graph(
            grid,
            background=0,
            connectivity=4,
            grouping="color_groups",
        )
        self.assertEqual(len(components.objects), 4)
        self.assertEqual(len(color_groups.objects), 2)
        self.assertEqual(tuple(item.index for item in color_groups.objects), (0, 1))
        self.assertEqual(color_groups.objects[0].colors, (1,))
        self.assertEqual(color_groups.objects[1].colors, (2,))
        self.assertEqual(
            color_groups,
            extract_scene_graph(
                grid,
                background=0,
                connectivity=4,
                grouping="color_groups",
            ),
        )

    def test_relation_free_scene_preserves_object_semantics(self) -> None:
        grid = as_grid(
            [
                [1, 0, 2, 0],
                [1, 0, 2, 2],
                [0, 0, 0, 0],
                [3, 3, 0, 4],
            ]
        )
        complete = extract_scene_graph(
            grid,
            background=0,
            connectivity=4,
            grouping="monochrome_components",
        )
        relation_free = extract_scene_graph(
            grid,
            background=0,
            connectivity=4,
            grouping="monochrome_components",
            include_relations=False,
        )
        self.assertEqual(relation_free.objects, complete.objects)
        self.assertEqual(relation_free.relations, ())
        node = CorrespondObjectsNode(
            "equivalence",
            ("relative_position", "shape", "size", "topology"),
            True,
        )
        self.assertEqual(
            object_correspondences(relation_free, node),
            object_correspondences(complete, node),
        )

    def test_correspondence_uses_shape_size_topology_and_relative_position(self) -> None:
        grid = as_grid(
            [
                [1, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        scene = extract_scene_graph(
            grid,
            background=0,
            connectivity=4,
            grouping="monochrome_components",
        )
        shape_classes = object_correspondences(
            scene,
            CorrespondObjectsNode("equivalence", ("shape", "size"), True),
        )
        self.assertEqual(len(shape_classes), 1)
        self.assertEqual(shape_classes[0].object_indices, (0, 1))
        positioned = object_correspondences(
            scene,
            CorrespondObjectsNode(
                "equivalence",
                ("relative_position", "shape", "size", "topology"),
                True,
            ),
        )
        self.assertEqual(len(positioned), 2)

    def test_scene_crop_has_explicit_asymmetric_canvas_and_node_trace(self) -> None:
        source = [
            [0, 8, 0, 8, 0],
            [0, 5, 5, 5, 0],
            [0, 5, 0, 5, 0],
            [0, 8, 0, 8, 0],
        ]
        expected = as_grid(
            [
                [8, 0, 8],
                [5, 5, 5],
                [5, 0, 5],
                [8, 0, 8],
            ]
        )
        program = _crop_program(padding=(1, 1, 0, 0), role="largest_area")
        execution = execute_object_code_program(program, as_grid(source))
        self.assertTrue(execution.ok)
        self.assertEqual(execution.output, expected)
        self.assertEqual(
            tuple(item.node_id for item in execution.node_trace),
            ("parse", "correspond", "select", "operate", "canvas", "render"),
        )
        self.assertEqual(execution.node_trace[-2].status, "ok")

    def test_scene_ast_round_trip_and_typed_node_hole(self) -> None:
        source = [
            [0, 0, 0, 0, 0],
            [0, 4, 4, 4, 0],
            [0, 4, 0, 4, 0],
            [0, 4, 4, 4, 0],
            [0, 0, 0, 0, 0],
        ]
        target = [[4, 4, 4], [4, 0, 4], [4, 4, 4]]
        task = _blind(source, target)
        program = _crop_program()
        reconstructed = object_code_program_from_json(program.to_json_dict())
        self.assertEqual(reconstructed, program)
        first = make_object_code_hypothesis(program, demo_exact=True)
        second = make_object_code_hypothesis(reconstructed, demo_exact=True)
        self.assertEqual(first.hypothesis_id, second.hypothesis_id)
        provisional = make_object_code_hypothesis(
            program,
            demo_exact=False,
            ast_holes=("ast.select.role",),
        )
        evaluation = evaluate_hypothesis(provisional, task)
        self.assertTrue(evaluation.demo_exact)
        self.assertFalse(evaluation.eligible)
        certificate, frontier = typed_repair_frontier(task, evaluation)
        self.assertEqual(certificate.recommended_action, "fill_ast_hole")
        self.assertEqual(certificate.affected_slots, ("ast.select.role",))
        self.assertTrue(frontier)

    def test_copy_count_arrange_and_compose_nodes_execute(self) -> None:
        grid = as_grid(
            [
                [1, 0, 0, 2],
                [0, 0, 0, 2],
                [3, 0, 0, 0],
            ]
        )
        parse = ParseObjectsNode(0, 4, "monochrome_components")
        none = CorrespondObjectsNode()
        all_objects = SelectObjectsNode("all")

        count = ScenePipelineProgram(
            parse,
            none,
            all_objects,
            ObjectOperationNode("count", "identity", "row", 0, "scene_objects", 7),
            CanvasNode("count_line", 0),
            RenderObjectsNode("solid"),
        )
        self.assertEqual(
            execute_object_code_program(count, grid).output,
            as_grid([[7, 7, 7]]),
        )

        arrange = ScenePipelineProgram(
            parse,
            none,
            all_objects,
            ObjectOperationNode("arrange", "identity", "row", 0),
            CanvasNode("tight", 0),
            RenderObjectsNode("objects"),
        )
        self.assertEqual(
            execute_object_code_program(arrange, grid).output,
            as_grid([[1, 2, 3], [0, 2, 0]]),
        )

        copy = ScenePipelineProgram(
            parse,
            none,
            SelectObjectsNode("smallest_area"),
            ObjectOperationNode("copy", "identity", "row", 0, "scene_objects"),
            CanvasNode("tight", 0),
            RenderObjectsNode("objects"),
        )
        self.assertEqual(
            execute_object_code_program(copy, grid).output,
            as_grid([[1, 1, 1]]),
        )

        compose = ScenePipelineProgram(
            parse,
            none,
            all_objects,
            ObjectOperationNode("compose"),
            CanvasNode("tight", 0),
            RenderObjectsNode("objects", "last"),
        )
        self.assertIsNotNone(execute_object_code_program(compose, grid).output)

    def test_scene_synthesis_reaches_crop_and_novel_frontier_is_mandatory(self) -> None:
        source = [
            [0, 2, 2, 2, 0, 0],
            [0, 2, 2, 2, 0, 8],
            [0, 2, 2, 2, 0, 8],
            [0, 0, 0, 0, 0, 0],
        ]
        target = [[8], [8]]
        task = _blind(source, target)
        synthesis = synthesize_object_code_programs(
            task,
            max_program_trials=20_000,
            max_exact_programs=8,
            max_near_misses=4,
        )
        self.assertTrue(
            any(
                isinstance(score.program, ScenePipelineProgram)
                for score in synthesis.exact_scores
            )
        )

        parent = make_object_code_hypothesis(
            ScenePipelineProgram(
                ParseObjectsNode(0, 4, "color_groups"),
                CorrespondObjectsNode(),
                SelectObjectsNode("smallest_area"),
                ObjectOperationNode("crop"),
                CanvasNode("bbox", 0, (0, 1, 0, 0)),
                RenderObjectsNode("source_crop"),
            ),
            demo_exact=False,
        )
        evaluation = evaluate_hypothesis(parent, task)
        _, frontier = typed_repair_frontier(task, evaluation)
        self.assertTrue(frontier)
        _, exhausted = typed_repair_frontier(
            task,
            evaluation,
            existing_programs=frontier,
        )
        self.assertEqual(exhausted, ())


if __name__ == "__main__":
    unittest.main()
