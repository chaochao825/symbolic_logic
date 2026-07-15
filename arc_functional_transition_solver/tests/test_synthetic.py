from __future__ import annotations

import hashlib
import unittest
from collections import Counter

from afts_arc.dsl import Program, execute_program
from afts_arc.panel import parse_panels
from afts_arc.relation import ContactStructuralStatus, parse_bbox_contacts
from afts_arc.search import (
    SearchConfig,
    bbox_contact_instruction_proposals,
    dsl_candidates,
    panel_lattice_periodic_instruction_proposals,
    search_programs,
)
from afts_arc.synthetic import generate_case, generate_suite


class SyntheticTests(unittest.TestCase):
    def test_suite_is_deterministic_and_keeps_oracle_metadata_out_of_blind_tasks(self) -> None:
        first = generate_suite(seed=3, tasks_per_family=1)
        second = generate_suite(seed=3, tasks_per_family=1)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 17)
        for case in first:
            blind_json = case.blind_task.to_json_dict()
            self.assertNotIn("family", blind_json)
            self.assertNotIn("generator_program", blind_json)
            self.assertNotIn("output", blind_json["test"][0])  # type: ignore[index]

    def test_generator_program_reproduces_every_output(self) -> None:
        for case in generate_suite(seed=1, tasks_per_family=1):
            for pair in (*case.task.train, *case.task.test):
                outcome = execute_program(case.generator_program, pair.input)
                self.assertTrue(outcome.ok, case.family)
                self.assertEqual(outcome.output, pair.output, case.family)

    def test_small_search_recovers_generator_or_equivalent_query_outputs(self) -> None:
        config = SearchConfig(max_depth=2, beam_width=64, max_instruction_options=64)
        for family in (
            "rotate90",
            "recolor",
            "rotate90_recolor",
            "crop_non_background",
            "crop_rotate90",
            "keep_largest_object",
            "scale_pixels_2x2",
            "tile_horizontal_2",
            "overlay_panel_grid",
            "overlay_panel_rotate90",
            "broadcast_panel_sequence_d4",
            "broadcast_panel_sequence_d4_flip_horizontal",
        ):
            case = generate_case(family=family, seed=17)
            result = search_programs(case.blind_task, config=config)
            outputs_by_index: dict[int, set[object]] = {
                index: set() for index in range(len(case.task.test))
            }
            for candidate in dsl_candidates(case.blind_task, search_result=result):
                outputs_by_index[candidate.test_index].add(candidate.output)
            self.assertTrue(
                all(
                    pair.output in outputs_by_index[index]
                    for index, pair in enumerate(case.task.test)
                ),
                family,
            )

    def test_panel_control_varies_separator_and_disambiguates_color_priority(self) -> None:
        case = generate_case(family="overlay_panel_grid", seed=17)
        pairs = (*case.task.train, *case.task.test)
        separators = [
            parse_panels(pair.input).hypotheses[0].separator_color for pair in pairs
        ]
        self.assertGreater(len(set(separators)), 1)
        for pair, separator in zip(pairs, separators):
            palette = tuple(color for color in range(1, 10) if color != separator)
            self.assertEqual(pair.output[0][0], max(palette))
            self.assertEqual(pair.output[0][1], min(palette))

    def test_panel_sequence_controls_vary_structure_and_retain_generator_programs(
        self,
    ) -> None:
        config = SearchConfig(max_depth=2, beam_width=64, max_instruction_options=64)
        for family in (
            "broadcast_panel_sequence_d4",
            "broadcast_panel_sequence_d4_flip_horizontal",
        ):
            with self.subTest(family=family):
                case = generate_case(family=family, seed=17)
                background = case.generator_program.instructions[0].arguments[
                    "background"
                ]
                axes: set[str] = set()
                panel_counts: set[int] = set()
                seed_positions: set[int] = set()
                for pair in (*case.task.train, *case.task.test):
                    eligible = [
                        hypothesis
                        for hypothesis in parse_panels(pair.input).hypotheses
                        if (
                            bool(hypothesis.row_separator_bands)
                            ^ bool(hypothesis.column_separator_bands)
                        )
                        and hypothesis.separator_color != background
                        and len(
                            {
                                (panel.bbox.height, panel.bbox.width)
                                for panel in hypothesis.panels
                            }
                        )
                        == 1
                    ]
                    self.assertEqual(len(eligible), 1)
                    selected = eligible[0]
                    axes.add(
                        "vertical"
                        if selected.row_separator_bands
                        else "horizontal"
                    )
                    panel_counts.add(len(selected.panels))
                    occupied = [
                        index
                        for index, panel in enumerate(selected.panels)
                        if any(
                            cell != background
                            for row in panel.content
                            for cell in row
                        )
                    ]
                    self.assertEqual(len(occupied), 1)
                    seed_positions.add(occupied[0])
                self.assertEqual(axes, {"horizontal", "vertical"})
                self.assertEqual(panel_counts, {2, 3, 4, 5})
                self.assertTrue(any(position > 0 for position in seed_positions))

                result = search_programs(case.blind_task, config=config)
                self.assertIn(
                    case.generator_program.program_id,
                    result.evaluated_program_ids,
                )
                self.assertIn(case.generator_program, result.exact_programs)
                one_step = search_programs(
                    case.blind_task,
                    config=SearchConfig(
                        max_depth=1,
                        beam_width=64,
                        max_instruction_options=64,
                    ),
                )
                if family == "broadcast_panel_sequence_d4":
                    self.assertEqual(
                        {
                            program.instructions[0].arguments["step"]
                            for program in one_step.exact_programs
                            if program.instructions[0].op
                            == "broadcast_panel_sequence_d4"
                        },
                        {case.generator_program.instructions[0].arguments["step"]},
                    )
                else:
                    self.assertFalse(one_step.exact_programs)
                    prefix = Program.create(
                        (case.generator_program.instructions[0],)
                    )
                    suffix = Program.create(
                        (case.generator_program.instructions[1],)
                    )
                    for pair in (*case.task.train, *case.task.test):
                        target = pair.output
                        self.assertNotEqual(
                            execute_program(prefix, pair.input).output,
                            target,
                        )
                        self.assertNotEqual(
                            execute_program(suffix, pair.input).output,
                            target,
                        )

    def test_v06_appends_new_families_without_changing_old_blind_semantics(
        self,
    ) -> None:
        cases = generate_suite(seed=0, tasks_per_family=3)
        self.assertEqual(len(cases), 51)
        pre_periodic_cases = cases[:39]
        pre_contact_cases = cases[:45]
        pre_periodic_digest = hashlib.sha256(
            "\n".join(
                f"{case.family}|{case.blind_task.task_id}"
                for case in pre_periodic_cases
            ).encode("ascii")
        ).hexdigest()
        self.assertEqual(
            pre_periodic_digest,
            "d5478bd1819202103e49940ecbd4be02f7deb437311723face08e6bdb26267be",
        )
        pre_contact_digest = hashlib.sha256(
            "\n".join(
                f"{case.family}|{case.blind_task.task_id}"
                for case in pre_contact_cases
            ).encode("ascii")
        ).hexdigest()
        self.assertEqual(
            pre_contact_digest,
            "d68083185d87ea33505e3c13cc15827066952272e27e24520c69ffe6000fb988",
        )
        self.assertTrue(
            all(
                not panel_lattice_periodic_instruction_proposals(
                    case.blind_task
                )
                for case in pre_periodic_cases
            )
        )
        self.assertTrue(
            all(
                not bbox_contact_instruction_proposals(case.blind_task)
                for case in pre_contact_cases
            )
        )

    def test_bbox_contact_families_retain_generator_programs(self) -> None:
        cases = generate_suite(seed=0, tasks_per_family=3)
        config = SearchConfig(max_depth=2, beam_width=64, max_instruction_options=64)
        for family in (
            "paint_bbox_contacts",
            "paint_bbox_contacts_rotate180",
        ):
            family_cases = [case for case in cases if case.family == family]
            self.assertEqual(len(family_cases), 3)
            self.assertEqual(
                {
                    case.generator_program.instructions[0].arguments["background"]
                    for case in family_cases
                },
                {0, 2, 3},
            )
            for case in family_cases:
                atomic = family == "paint_bbox_contacts"
                result = search_programs(
                    case.blind_task,
                    config=(
                        SearchConfig(
                            max_depth=1,
                            beam_width=64,
                            max_instruction_options=64,
                        )
                        if atomic
                        else config
                    ),
                )
                self.assertIn(
                    case.generator_program.instructions[0],
                    result.instruction_options,
                )
                self.assertIn(
                    case.generator_program.program_id,
                    result.evaluated_program_ids,
                )
                self.assertIn(case.generator_program, result.exact_programs)
                self.assertEqual(
                    result.bbox_contact_admissible_binding_count,
                    1,
                )
                self.assertEqual(result.bbox_contact_action_trial_count, 1)
                self.assertEqual(result.bbox_contact_demo_execution_count, 3)
                if not atomic:
                    one_step = search_programs(
                        case.blind_task,
                        config=SearchConfig(
                            max_depth=1,
                            beam_width=64,
                            max_instruction_options=64,
                        ),
                    )
                    self.assertFalse(one_step.exact_programs)
                    prefix = Program.create(
                        (case.generator_program.instructions[0],)
                    )
                    suffix = Program.create(
                        (case.generator_program.instructions[1],)
                    )
                    for pair in (*case.task.train, *case.task.test):
                        self.assertNotEqual(
                            execute_program(prefix, pair.input).output,
                            pair.output,
                        )
                        self.assertNotEqual(
                            execute_program(suffix, pair.input).output,
                            pair.output,
                        )

    def test_bbox_contact_controls_cover_registered_structural_diversity(self) -> None:
        cases = generate_suite(seed=0, tasks_per_family=3)
        contact_cases = [
            case
            for case in cases
            if case.family
            in {"paint_bbox_contacts", "paint_bbox_contacts_rotate180"}
        ]
        backgrounds: set[int] = set()
        anchor_colors: set[int] = set()
        anchor_shapes: set[tuple[int, int]] = set()
        anchor_positions: set[tuple[int, int]] = set()
        directions: set[str] = set()
        gaps: set[int] = set()
        marker_counts: set[int] = set()
        repeated_marker_color = False
        multiple_markers_on_one_side = False
        for case in contact_cases:
            background = case.generator_program.instructions[0].arguments[
                "background"
            ]
            backgrounds.add(background)
            for pair in (*case.task.train, *case.task.test):
                bundle = parse_bbox_contacts(pair.input, background=background)
                complete = [
                    item
                    for item in bundle.hypotheses
                    if item.structural_status is ContactStructuralStatus.COMPLETE
                ]
                self.assertEqual(len(complete), 1)
                hypothesis = complete[0]
                anchor_colors.add(hypothesis.anchor_color)
                anchor_shapes.add(
                    (hypothesis.anchor_bbox.height, hypothesis.anchor_bbox.width)
                )
                anchor_positions.add(
                    (hypothesis.anchor_bbox.top, hypothesis.anchor_bbox.left)
                )
                marker_counts.add(len(hypothesis.relations))
                direction_counts = Counter(
                    relation.direction.value for relation in hypothesis.relations
                )
                directions.update(direction_counts)
                multiple_markers_on_one_side |= max(direction_counts.values()) >= 2
                gaps.update(relation.gap for relation in hypothesis.relations)
                marker_colors = [
                    pair.input[relation.marker_coordinate[0]][
                        relation.marker_coordinate[1]
                    ]
                    for relation in hypothesis.relations
                ]
                repeated_marker_color |= len(set(marker_colors)) < len(marker_colors)
        self.assertEqual(backgrounds, {0, 2, 3})
        self.assertGreaterEqual(len(anchor_colors), 3)
        self.assertGreaterEqual(len(anchor_shapes), 3)
        self.assertGreaterEqual(len(anchor_positions), 2)
        self.assertEqual(directions, {"left", "right", "up", "down"})
        self.assertEqual(gaps, {0, 1, 2})
        self.assertEqual(marker_counts, {3, 4, 5, 6})
        self.assertTrue(repeated_marker_color)
        self.assertTrue(multiple_markers_on_one_side)

    def test_periodic_lattice_formal_tasks_cover_registered_parameters(self) -> None:
        cases = generate_suite(seed=0, tasks_per_family=3)
        config = SearchConfig(max_depth=2, beam_width=64, max_instruction_options=64)
        for family in (
            "broadcast_panel_lattice_periodic",
            "broadcast_panel_lattice_periodic_rotate180",
        ):
            family_cases = [case for case in cases if case.family == family]
            self.assertEqual(len(family_cases), 3)
            self.assertEqual(
                {
                    (
                        case.generator_program.instructions[0].arguments[
                            "row_period"
                        ],
                        case.generator_program.instructions[0].arguments[
                            "column_period"
                        ],
                    )
                    for case in family_cases
                },
                {(2, 2), (2, 3), (3, 2)},
            )
            self.assertEqual(
                {
                    case.generator_program.instructions[0].arguments["background"]
                    for case in family_cases
                },
                {0, 2, 3},
            )
            for case in family_cases:
                instruction = case.generator_program.instructions[0]
                atomic = family == "broadcast_panel_lattice_periodic"
                one_step_config = SearchConfig(
                    max_depth=1,
                    beam_width=64,
                    max_instruction_options=64,
                )
                result = search_programs(
                    case.blind_task,
                    config=one_step_config if atomic else config,
                )
                self.assertIn(instruction, result.instruction_options)
                self.assertIn(
                    case.generator_program.program_id,
                    result.evaluated_program_ids,
                )
                self.assertIn(case.generator_program, result.exact_programs)
                one_step = (
                    result
                    if atomic
                    else search_programs(
                        case.blind_task,
                        config=one_step_config,
                    )
                )
                if atomic:
                    periodic_exact = {
                        (
                            program.instructions[0].arguments["background"],
                            program.instructions[0].arguments["row_period"],
                            program.instructions[0].arguments["column_period"],
                        )
                        for program in one_step.exact_programs
                        if program.instructions[0].op
                        == "broadcast_panel_lattice_periodic"
                    }
                    self.assertEqual(
                        periodic_exact,
                        {
                            (
                                instruction.arguments["background"],
                                instruction.arguments["row_period"],
                                instruction.arguments["column_period"],
                            )
                        },
                    )
                else:
                    self.assertFalse(one_step.exact_programs)
                    prefix = Program.create((case.generator_program.instructions[0],))
                    suffix = Program.create((case.generator_program.instructions[1],))
                    for pair in (*case.task.train, *case.task.test):
                        target = pair.output
                        self.assertNotEqual(
                            execute_program(prefix, pair.input).output,
                            target,
                        )
                        self.assertNotEqual(
                            execute_program(suffix, pair.input).output,
                            target,
                        )

    def test_periodic_lattice_controls_cover_raggedness_and_search_gates(
        self,
    ) -> None:
        for family in (
            "broadcast_panel_lattice_periodic",
            "broadcast_panel_lattice_periodic_rotate180",
        ):
            with self.subTest(family=family):
                case = generate_case(family=family, seed=17)
                instruction = case.generator_program.instructions[0]
                background = instruction.arguments["background"]
                row_period = instruction.arguments["row_period"]
                column_period = instruction.arguments["column_period"]
                modes: set[str] = set()
                phases: set[tuple[int, int]] = set()
                separator_colors: set[int] = set()
                separator_thicknesses: set[int] = set()
                selected_partial = False
                unselected_partial = False
                for pair in (*case.task.train, *case.task.test):
                    eligible = [
                        hypothesis
                        for hypothesis in parse_panels(pair.input).hypotheses
                        if hypothesis.row_separator_bands
                        and hypothesis.column_separator_bands
                        and hypothesis.separator_color != background
                    ]
                    self.assertEqual(len(eligible), 1)
                    selected = eligible[0]
                    separator_colors.add(selected.separator_color)
                    separator_thicknesses.update(
                        band.end - band.start + 1
                        for band in (
                            *selected.row_separator_bands,
                            *selected.column_separator_bands,
                        )
                    )
                    nominal_height = max(panel.bbox.height for panel in selected.panels)
                    nominal_width = max(panel.bbox.width for panel in selected.panels)
                    occupied = [
                        panel
                        for panel in selected.panels
                        if any(
                            cell != background
                            for row in panel.content
                            for cell in row
                        )
                    ]
                    self.assertEqual(len(occupied), 1)
                    source = occupied[0]
                    self.assertEqual(
                        (source.bbox.height, source.bbox.width),
                        (nominal_height, nominal_width),
                    )
                    phases.add(
                        (
                            source.row_index % row_period,
                            source.column_index % column_period,
                        )
                    )
                    periodic = execute_program(
                        Program.create((instruction,)), pair.input
                    )
                    self.assertTrue(periodic.ok)
                    self.assertIsNotNone(periodic.output)
                    last_row = max(panel.row_index for panel in selected.panels)
                    last_column = max(panel.column_index for panel in selected.panels)
                    bottom = any(
                        panel.bbox.height < nominal_height
                        for panel in selected.panels
                    )
                    right = any(
                        panel.bbox.width < nominal_width
                        for panel in selected.panels
                    )
                    modes.add(
                        "both"
                        if bottom and right
                        else "bottom"
                        if bottom
                        else "right"
                        if right
                        else "uniform"
                    )
                    for panel in selected.panels:
                        if panel.bbox.height < nominal_height:
                            self.assertEqual(panel.row_index, last_row)
                        if panel.bbox.width < nominal_width:
                            self.assertEqual(panel.column_index, last_column)
                        if (
                            panel.bbox.height == nominal_height
                            and panel.bbox.width == nominal_width
                        ):
                            continue
                        is_selected = (
                            (panel.row_index - source.row_index) % row_period == 0
                            and (panel.column_index - source.column_index)
                            % column_period
                            == 0
                        )
                        actual = tuple(
                            tuple(
                                periodic.output[row][
                                    panel.bbox.left : panel.bbox.right + 1
                                ]
                            )
                            for row in range(
                                panel.bbox.top, panel.bbox.bottom + 1
                            )
                        )
                        expected = (
                            tuple(
                                row[: panel.bbox.width]
                                for row in source.content[: panel.bbox.height]
                            )
                            if is_selected
                            else panel.content
                        )
                        self.assertEqual(actual, expected)
                        selected_partial |= is_selected
                        unselected_partial |= not is_selected
                self.assertEqual(modes, {"uniform", "bottom", "right", "both"})
                self.assertGreater(len(phases), 1)
                self.assertGreater(len(separator_colors), 1)
                self.assertEqual(separator_thicknesses, {1, 2})
                self.assertTrue(selected_partial)
                self.assertTrue(unselected_partial)


if __name__ == "__main__":
    unittest.main()
