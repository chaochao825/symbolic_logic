from __future__ import annotations

import pytest

from afts_arc.blind import BlindTask
from afts_arc.experiment_safety import canonical_sha256
from afts_arc.grid import Grid, as_grid
from afts_arc.hybrid.scene_graph import ParseObjectsNode, extract_scene_graph
from afts_arc.object_program_workspace import (
    OBJECT_REMATCH_SLOT,
    NaryObjectCorrespondenceNode,
    ObjectEffectNode,
    ObjectRematchProgram,
    base_prefix_programs,
    compile_object_rematch_certificate,
    enumerate_object_rematch_programs,
    execute_object_rematch,
    infer_effect_assignment,
    object_rematch_near_miss_quality,
    object_rematch_variants,
    posterior_assignment_distribution,
    rank_object_rematch_variants,
    relation_profiles,
    score_object_rematch_program,
    select_object_ids,
    synthesize_exact_object_rematch_programs,
)
from afts_arc.task import ARCPair


def _paint(
    height: int,
    width: int,
    objects: list[tuple[int, tuple[tuple[int, int], ...]]],
) -> Grid:
    grid = [[0 for _ in range(width)] for _ in range(height)]
    for color, cells in objects:
        for row, column in cells:
            grid[row][column] = color
    return as_grid(grid)


def _recolor_cells(grid: Grid, cells: set[tuple[int, int]], color: int) -> Grid:
    output = [list(row) for row in grid]
    for row, column in cells:
        output[row][column] = color
    return as_grid(output)


def _workspace_fixture() -> tuple[
    BlindTask,
    ObjectRematchProgram,
    ObjectRematchProgram,
    ObjectRematchProgram,
    tuple[Grid, ...],
]:
    first = _paint(
        9,
        9,
        [
            (2, ((2, 2), (3, 2), (3, 3))),
            (3, ((5, 5),)),
            (4, ((0, 1), (0, 2))),
            (4, ((8, 6), (8, 7))),
        ],
    )
    first_gold = _recolor_cells(
        first,
        {(2, 2), (3, 2), (3, 3), (5, 5)},
        8,
    )
    second = _paint(
        10,
        10,
        [
            (5, ((2, 2), (2, 3), (3, 2), (3, 3))),
            (6, ((6, 6), (7, 6))),
            (4, ((0, 1),)),
            (4, ((0, 5),)),
            (4, ((9, 8),)),
        ],
    )
    second_gold = _recolor_cells(
        second,
        {(2, 2), (2, 3), (3, 2), (3, 3), (6, 6), (7, 6)},
        8,
    )
    query = _paint(
        9,
        9,
        [
            (2, ((3, 3),)),
            (3, ((5, 5),)),
            (4, ((0, 1), (0, 2))),
        ],
    )
    query_gold = _recolor_cells(query, {(3, 3), (5, 5)}, 8)
    task = BlindTask.from_observations(
        train=(ARCPair(first, first_gold), ARCPair(second, second_gold)),
        test_inputs=(query,),
    )
    parse = ParseObjectsNode(0, 4, "monochrome_components")
    effect = ObjectEffectNode("recolor_component", 8)
    parent = ObjectRematchProgram(
        parse,
        NaryObjectCorrespondenceNode("max_area"),
        effect,
    )
    interior = ObjectRematchProgram(
        parse,
        NaryObjectCorrespondenceNode("interior"),
        effect,
    )
    rarest = ObjectRematchProgram(
        parse,
        NaryObjectCorrespondenceNode("rarest_shape"),
        effect,
    )
    return task, parent, interior, rarest, (query_gold,)


def test_multi_parse_profiles_and_nary_selectors_are_deterministic() -> None:
    grid = as_grid(
        [
            [0, 2, 0, 0],
            [0, 0, 2, 0],
            [3, 0, 0, 3],
            [0, 0, 0, 0],
        ]
    )
    scene4 = extract_scene_graph(
        grid,
        background=0,
        connectivity=4,
        grouping="monochrome_components",
    )
    scene8 = extract_scene_graph(
        grid,
        background=0,
        connectivity=8,
        grouping="monochrome_components",
    )

    assert len(scene4.objects) == 4
    assert len(scene8.objects) == 3
    assert relation_profiles(scene4) == relation_profiles(scene4)
    assert select_object_ids(
        scene4, NaryObjectCorrespondenceNode("rarest_shape")
    )
    assert select_object_ids(
        scene4, NaryObjectCorrespondenceNode("most_related")
    )


def test_program_round_trip_execution_and_complete_object_assignment() -> None:
    task, parent, interior, _, query_gold = _workspace_fixture()

    assert ObjectRematchProgram.from_json_dict(interior.to_json_dict()) == interior
    execution = execute_object_rematch(interior, task.test_inputs[0])

    assert execution.ok
    assert execution.output == query_gold[0]
    assert tuple(node.node_id for node in execution.node_trace) == (
        "parse",
        "relation_graph",
        "correspond",
        "effect",
        "render",
    )
    assert infer_effect_assignment(
        interior, task.test_inputs[0], query_gold[0]
    ) == execution.selected_object_ids
    assert execute_object_rematch(parent, task.train[0].input).output != (
        task.train[0].output
    )


def test_synthesis_finds_demo_exact_relational_selectors() -> None:
    task, _, interior, rarest, _ = _workspace_fixture()
    exact_ids = {
        program.program_id
        for program in synthesize_exact_object_rematch_programs(task)
    }

    assert interior.program_id in exact_ids
    assert rarest.program_id in exact_ids
    assert all(
        program.correspond.selector in {
            "interior",
            "rarest_shape",
            "rarest_size",
            "rarest_topology",
        }
        or program.parse != interior.parse
        or program.effect != interior.effect
        for program in synthesize_exact_object_rematch_programs(task)
    )


def test_joint_posterior_preserves_assignment_and_rejects_partial_pixels() -> None:
    task, _, interior, rarest, query_gold = _workspace_fixture()
    query = task.test_inputs[0]
    rarest_output = execute_object_rematch(rarest, query).output
    assert rarest_output is not None
    partial = _recolor_cells(query, {(0, 1)}, 8)
    distribution = posterior_assignment_distribution(
        interior,
        query,
        (
            (query_gold[0], 8),
            (rarest_output, 2),
            (partial, 3),
        ),
    )
    exact_assignment = execute_object_rematch(interior, query).selected_object_ids

    assert distribution.total_weight == 13
    assert distribution.represented_weight == 10
    assert distribution.rejected_weight == 3
    assert distribution.weight_for(exact_assignment) == 8
    assert distribution.distribution_id == canonical_sha256(
        {
            key: value
            for key, value in distribution.to_json_dict().items()
            if key != "distribution_id"
        }
    )


def test_certificate_compiles_residual_to_one_existing_typed_slot() -> None:
    task, parent, _, _, _ = _workspace_fixture()
    parent_views = []
    for pair in task.train:
        parent_output = execute_object_rematch(parent, pair.input).output
        assert parent_output is not None
        parent_views.append(((pair.output, 3), (parent_output, 1)))

    certificate = compile_object_rematch_certificate(
        task_id="fixture",
        task=task,
        parent=parent,
        mode="joint_posterior",
        posterior_views=tuple(parent_views),
    )
    quality = object_rematch_near_miss_quality(parent, task)

    assert quality.eligible
    assert quality.delta_precision == 1.0
    assert quality.delta_recall > 0.5
    assert certificate.action == "object_rematch"
    assert certificate.affected_slots == (OBJECT_REMATCH_SLOT,)
    assert certificate.evidence["pixel_marginal_used"] is False
    assert certificate.certificate_id == canonical_sha256(
        {
            key: value
            for key, value in certificate.to_json_dict().items()
            if key != "certificate_id"
        }
    )


def test_frontier_requires_novel_single_slot_programs() -> None:
    task, parent, interior, _, _ = _workspace_fixture()
    programs = enumerate_object_rematch_programs(task)
    base = base_prefix_programs(programs)
    edits = object_rematch_variants(
        parent,
        candidate_programs=programs,
        existing_program_ids=tuple(program.program_id for program in base),
    )

    assert edits
    assert any(edit.program.program_id == interior.program_id for edit in edits)
    assert all(edit.slot == OBJECT_REMATCH_SLOT for edit in edits)
    assert all(edit.program.parse == parent.parse for edit in edits)
    assert all(edit.program.effect == parent.effect for edit in edits)
    assert not object_rematch_variants(
        parent,
        candidate_programs=(interior,),
        existing_program_ids=(interior.program_id,),
    )


def test_visual_assignment_mass_ranks_demo_exact_query_behavior() -> None:
    task, parent, interior, rarest, query_gold = _workspace_fixture()
    rarest_output = execute_object_rematch(rarest, task.test_inputs[0]).output
    assert rarest_output is not None
    certificate = compile_object_rematch_certificate(
        task_id="fixture",
        task=task,
        parent=parent,
        mode="residual_only",
        posterior_views=(),
    )
    programs = enumerate_object_rematch_programs(task)
    base = base_prefix_programs(programs)
    ranked = rank_object_rematch_variants(
        task=task,
        parent=parent,
        certificate=certificate,
        candidate_programs=programs,
        existing_program_ids=tuple(program.program_id for program in base),
        query_index=0,
        query_posterior=((query_gold[0], 8), (rarest_output, 2)),
        use_posterior=True,
    )

    assert ranked[0].demo_exact
    assert ranked[0].posterior_assignment_weight == 8
    assert ranked[0].query_output == query_gold[0]
    assert ranked[0].edit.program.program_id not in {
        program.program_id for program in base
    }


def test_certificate_rejects_nonrepresentable_or_exact_failures() -> None:
    task, _, interior, _, _ = _workspace_fixture()

    with pytest.raises(ValueError, match="exact parent"):
        compile_object_rematch_certificate(
            task_id="fixture",
            task=task,
            parent=interior,
            mode="residual_only",
            posterior_views=(),
        )

    with pytest.raises(ValueError, match="cannot receive posterior"):
        compile_object_rematch_certificate(
            task_id="fixture",
            task=task,
            parent=ObjectRematchProgram(
                interior.parse,
                NaryObjectCorrespondenceNode("max_area"),
                interior.effect,
            ),
            mode="residual_only",
            posterior_views=(((task.train[0].output, 1),),),
        )


def test_same_canvas_program_scores_shape_changing_demo_without_raising() -> None:
    task = BlindTask.from_observations(
        train=(ARCPair(as_grid([[2, 0], [0, 0]]), as_grid([[2]])),),
        test_inputs=(as_grid([[2, 0], [0, 0]]),),
    )
    program = ObjectRematchProgram(
        ParseObjectsNode(0, 4, "monochrome_components"),
        NaryObjectCorrespondenceNode("all"),
        ObjectEffectNode("recolor_component", 8),
    )
    score = score_object_rematch_program(program, task)

    assert score.execution_valid
    assert not score.all_demo_exact
    assert score.mismatch_count == 4
