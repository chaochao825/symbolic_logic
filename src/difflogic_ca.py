"""Execute the hard DigitalJS circuits published with Google DiffLogic-CA.

The companion article exposes trained, hard Boolean circuits as DigitalJS
JSON.  This module is a small, dependency-light adapter for those artifacts:

* downloads are pinned to one immutable commit and checked with SHA-256;
* DigitalJS devices are compiled into a topologically ordered Boolean DAG;
* ``Button.order`` and ``Output.order`` define the public tensor layout; and
* a synchronous 3x3 cellular-automaton update uses spatial-major patch order
  ``order = f * C + c`` (row-major spatial index ``f`` and channel ``c``).

It is an inference adapter only.  It does not implement, invoke, or claim to
reproduce the differentiable JAX training procedure from the paper/notebook.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
from urllib.request import urlopen

import numpy as np


GOOGLE_DIFFLOGIC_CA_COMMIT = "4c0246d9f7a2912cb7201f6bfe5fcda0fe373904"
GOOGLE_DIFFLOGIC_CA_RAW_ROOT = (
    "https://raw.githubusercontent.com/google-research/self-organising-systems/"
    + GOOGLE_DIFFLOGIC_CA_COMMIT
    + "/difflogic-ca"
)


@dataclass(frozen=True)
class GoogleDiffLogicCAArtifact:
    """Immutable metadata for one JSON file on the article's pages branch."""

    filename: str
    sha256: str

    @property
    def url(self) -> str:
        return GOOGLE_DIFFLOGIC_CA_RAW_ROOT + "/" + self.filename


GOOGLE_DIFFLOGIC_CA_ARTIFACTS = {
    "gol_circuit.json": GoogleDiffLogicCAArtifact(
        filename="gol_circuit.json",
        sha256="82d59f04ca8164ba31fdb12314ee147dec3c8a073b632aeced0ef1581acfecba",
    ),
    "checkerboard.json": GoogleDiffLogicCAArtifact(
        filename="checkerboard.json",
        sha256="8a22e6e1d4d2f4b284ea4d2818cfb87a939df82f1d318c101f42a8d18a134c58",
    ),
    "all_gates.json": GoogleDiffLogicCAArtifact(
        filename="all_gates.json",
        sha256="cff964da24b1cb4e787497a1c71f34ea9ab4a2d50729f9d0823f48be4110a4ca",
    ),
}


class ArtifactIntegrityError(ValueError):
    """Raised when downloaded bytes do not match their expected digest."""


class DigitalJSCircuitFormatError(ValueError):
    """Raised when a DigitalJS object cannot be compiled as a Boolean DAG."""


PathLike = Union[str, os.PathLike]


def sha256_hexdigest(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest for raw artifact bytes."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    return hashlib.sha256(payload).hexdigest()


def _validate_json_object(payload: bytes) -> Mapping[str, Any]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DigitalJSCircuitFormatError("artifact is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, Mapping):
        raise DigitalJSCircuitFormatError("the JSON root must be an object")
    return decoded


def download_json_with_sha256(
    url: str,
    destination: PathLike,
    expected_sha256: str,
    *,
    timeout: float = 60.0,
    max_bytes: int = 32 * 1024 * 1024,
    overwrite: bool = False,
    opener: Optional[Any] = None,
) -> Path:
    """Download one JSON object atomically after checking its SHA-256.

    ``opener`` is injectable so the integrity path can be tested without a
    network connection.  Existing files are accepted only when their digest
    already matches; otherwise ``overwrite=True`` is required.
    """

    if not isinstance(url, str) or not url:
        raise ValueError("url must be a nonempty string")
    expected_sha256 = str(expected_sha256).lower()
    if len(expected_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha256):
        raise ValueError("expected_sha256 must contain 64 hexadecimal characters")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")

    target = Path(destination)
    if target.exists():
        existing = target.read_bytes()
        if sha256_hexdigest(existing) == expected_sha256:
            _validate_json_object(existing)
            return target
        if not overwrite:
            raise ArtifactIntegrityError(
                "destination exists but does not have the expected SHA-256; "
                "pass overwrite=True to replace it"
            )

    request_open = urlopen if opener is None else opener
    with request_open(url, timeout=timeout) as response:
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ArtifactIntegrityError("download exceeds max_bytes")
    actual_sha256 = sha256_hexdigest(payload)
    if actual_sha256 != expected_sha256:
        raise ArtifactIntegrityError(
            "SHA-256 mismatch: expected %s, received %s" % (expected_sha256, actual_sha256)
        )
    _validate_json_object(payload)

    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=target.name + ".",
        suffix=".tmp",
        dir=str(target.parent),
    )
    try:
        with os.fdopen(file_descriptor, "wb") as output_file:
            output_file.write(payload)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_name, str(target))
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return target


def download_google_difflogic_ca_json(
    destination: PathLike,
    *,
    artifact: str = "gol_circuit.json",
    timeout: float = 60.0,
    overwrite: bool = False,
) -> Path:
    """Download a commit-pinned Google DiffLogic-CA DigitalJS artifact."""

    if not artifact.endswith(".json"):
        artifact = artifact + ".json"
    if artifact not in GOOGLE_DIFFLOGIC_CA_ARTIFACTS:
        choices = ", ".join(sorted(GOOGLE_DIFFLOGIC_CA_ARTIFACTS))
        raise ValueError("unknown artifact %r; choose one of %s" % (artifact, choices))
    metadata = GOOGLE_DIFFLOGIC_CA_ARTIFACTS[artifact]
    return download_json_with_sha256(
        metadata.url,
        destination,
        metadata.sha256,
        timeout=timeout,
        overwrite=overwrite,
    )


# Canonical names are those emitted by the article artifacts.  The additional
# A/B and complemented pass-through operations complete the 16 two-input
# Boolean functions used by DiffLogic networks and make the adapter robust to
# unpruned exports.
_TYPE_ALIASES = {
    "button": "Button",
    "input": "Button",
    "output": "Output",
    "false": "False",
    "true": "True",
    "and": "And",
    "or": "Or",
    "xor": "Xor",
    "xnor": "Xnor",
    "nand": "Nand",
    "nor": "Nor",
    "not": "Not",
    "aandnotb": "AAndNotB",
    "notaandb": "NotAAndB",
    "aornotb": "AOrNotB",
    "notaorb": "NotAOrB",
    "a": "A",
    "b": "B",
    "nota": "NotA",
    "notb": "NotB",
    "buffer": "Buffer",
}

_CONSTANT_TYPES = frozenset(("False", "True"))
_UNARY_TYPES = frozenset(("Not", "Buffer"))
_BINARY_TYPES = frozenset(
    (
        "And",
        "Or",
        "Xor",
        "Xnor",
        "Nand",
        "Nor",
        "AAndNotB",
        "NotAAndB",
        "AOrNotB",
        "NotAOrB",
        "A",
        "B",
        "NotA",
        "NotB",
    )
)


def _canonical_type(device_type: Any) -> str:
    if not isinstance(device_type, str):
        raise DigitalJSCircuitFormatError("every device needs a string type")
    canonical = _TYPE_ALIASES.get(device_type.lower())
    if canonical is None:
        raise DigitalJSCircuitFormatError("unsupported DigitalJS device type %r" % device_type)
    return canonical


def _ordered_device_order(specification: Mapping[str, Any], device_id: str) -> int:
    order = specification.get("order")
    if isinstance(order, bool) or not isinstance(order, int) or order < 0:
        raise DigitalJSCircuitFormatError(
            "ordered device %r needs a nonnegative integer order" % device_id
        )
    return order


def _binary_array(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    try:
        valid = bool(np.all((array == 0) | (array == 1)))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("%s must contain only Boolean or binary values" % name)
    return array.astype(np.bool_, copy=False)


def _required_ports(device_type: str) -> Tuple[str, ...]:
    if device_type == "Output" or device_type in _UNARY_TYPES:
        return ("in",)
    if device_type in _BINARY_TYPES:
        return ("in1", "in2")
    return ()


def _evaluate_gate(device_type: str, inputs: Sequence[np.ndarray]) -> np.ndarray:
    if device_type == "Not":
        return np.logical_not(inputs[0])
    if device_type == "Buffer":
        return inputs[0]
    a, b = inputs
    if device_type == "And":
        return np.logical_and(a, b)
    if device_type == "Or":
        return np.logical_or(a, b)
    if device_type == "Xor":
        return np.logical_xor(a, b)
    if device_type == "Xnor":
        return np.logical_not(np.logical_xor(a, b))
    if device_type == "Nand":
        return np.logical_not(np.logical_and(a, b))
    if device_type == "Nor":
        return np.logical_not(np.logical_or(a, b))
    if device_type == "AAndNotB":
        return np.logical_and(a, np.logical_not(b))
    if device_type == "NotAAndB":
        return np.logical_and(np.logical_not(a), b)
    if device_type == "AOrNotB":
        return np.logical_or(a, np.logical_not(b))
    if device_type == "NotAOrB":
        return np.logical_or(np.logical_not(a), b)
    if device_type == "A":
        return a
    if device_type == "B":
        return b
    if device_type == "NotA":
        return np.logical_not(a)
    if device_type == "NotB":
        return np.logical_not(b)
    raise AssertionError("uncompiled gate type %r" % device_type)


@dataclass(frozen=True)
class _CompiledDevice:
    device_id: str
    device_type: str
    sources: Tuple[str, ...]


class DigitalJSBooleanCircuit:
    """A parsed DigitalJS Boolean circuit with batch NumPy evaluation."""

    def __init__(self, document: Mapping[str, Any]):
        if not isinstance(document, Mapping):
            raise DigitalJSCircuitFormatError("DigitalJS document must be a mapping")
        raw_devices = document.get("devices")
        raw_connectors = document.get("connectors")
        if not isinstance(raw_devices, Mapping):
            raise DigitalJSCircuitFormatError("DigitalJS document needs a devices object")
        if not isinstance(raw_connectors, list):
            raise DigitalJSCircuitFormatError("DigitalJS document needs a connectors list")
        subcircuits = document.get("subcircuits", {})
        if subcircuits:
            raise DigitalJSCircuitFormatError("nested DigitalJS subcircuits are not supported")

        specifications: Dict[str, Mapping[str, Any]] = {}
        device_types: Dict[str, str] = {}
        for raw_id, raw_specification in raw_devices.items():
            device_id = str(raw_id)
            if not isinstance(raw_specification, Mapping):
                raise DigitalJSCircuitFormatError("device %r must be an object" % device_id)
            specifications[device_id] = raw_specification
            device_types[device_id] = _canonical_type(raw_specification.get("type"))

        incoming: Dict[str, Dict[str, str]] = {device_id: {} for device_id in specifications}
        seen_connectors = set()
        for index, connector in enumerate(raw_connectors):
            if not isinstance(connector, Mapping):
                raise DigitalJSCircuitFormatError("connector %d must be an object" % index)
            source_endpoint = connector.get("from")
            target_endpoint = connector.get("to")
            if not isinstance(source_endpoint, Mapping) or not isinstance(target_endpoint, Mapping):
                raise DigitalJSCircuitFormatError("connector %d needs from/to objects" % index)
            source_id = str(source_endpoint.get("id"))
            target_id = str(target_endpoint.get("id"))
            target_port = target_endpoint.get("port")
            if source_id not in specifications or target_id not in specifications:
                raise DigitalJSCircuitFormatError("connector %d references an unknown device" % index)
            source_port = source_endpoint.get("port", "out")
            if source_port != "out":
                raise DigitalJSCircuitFormatError("connector %d uses an unsupported source port" % index)
            if not isinstance(target_port, str):
                raise DigitalJSCircuitFormatError("connector %d needs a string target port" % index)
            connector_key = (source_id, source_port, target_id, target_port)
            if connector_key in seen_connectors:
                continue
            seen_connectors.add(connector_key)
            if target_port in incoming[target_id] and incoming[target_id][target_port] != source_id:
                raise DigitalJSCircuitFormatError(
                    "device %r has multiple drivers for port %r" % (target_id, target_port)
                )
            incoming[target_id][target_port] = source_id

        dependencies: Dict[str, Tuple[str, ...]] = {}
        for device_id, device_type in device_types.items():
            required = _required_ports(device_type)
            provided = incoming[device_id]
            missing = [port for port in required if port not in provided]
            unexpected = [port for port in provided if port not in required]
            if missing:
                raise DigitalJSCircuitFormatError(
                    "device %r is missing input port(s): %s" % (device_id, ", ".join(missing))
                )
            if unexpected:
                raise DigitalJSCircuitFormatError(
                    "device %r has unsupported input port(s): %s"
                    % (device_id, ", ".join(sorted(unexpected)))
                )
            dependencies[device_id] = tuple(provided[port] for port in required)

        topological_ids = self._topological_order(dependencies)
        self._compiled = tuple(
            _CompiledDevice(device_id, device_types[device_id], dependencies[device_id])
            for device_id in topological_ids
        )
        type_counts: Dict[str, int] = {}
        for device_type in device_types.values():
            type_counts[device_type] = type_counts.get(device_type, 0) + 1
        self._device_type_count_items = tuple(sorted(type_counts.items()))
        self._raw_connector_count = len(raw_connectors)
        self._unique_connector_count = len(seen_connectors)

        input_pairs = []
        output_pairs = []
        for device_id, device_type in device_types.items():
            if device_type == "Button":
                input_pairs.append((_ordered_device_order(specifications[device_id], device_id), device_id))
            elif device_type == "Output":
                output_pairs.append((_ordered_device_order(specifications[device_id], device_id), device_id))
        self._validate_unique_orders(input_pairs, "Button")
        self._validate_unique_orders(output_pairs, "Output")
        input_pairs.sort()
        output_pairs.sort()
        self._input_pairs = tuple(input_pairs)
        self._output_pairs = tuple(output_pairs)
        self._input_order_by_id = {device_id: order for order, device_id in input_pairs}

        depths: Dict[str, int] = {}
        for device in self._compiled:
            if device.device_type == "Button":
                depths[device.device_id] = 0
            elif device.device_type in _CONSTANT_TYPES:
                depths[device.device_id] = 1
            elif device.device_type == "Output":
                depths[device.device_id] = depths[device.sources[0]]
            else:
                depths[device.device_id] = 1 + max(depths[source] for source in device.sources)
        if self._output_pairs:
            self._critical_depth = max(depths[device_id] for _, device_id in self._output_pairs)
        else:
            self._critical_depth = max(depths.values(), default=0)

    @staticmethod
    def _validate_unique_orders(pairs: Sequence[Tuple[int, str]], kind: str) -> None:
        orders = [order for order, _ in pairs]
        if len(set(orders)) != len(orders):
            raise DigitalJSCircuitFormatError("%s devices must have unique order values" % kind)

    @staticmethod
    def _topological_order(dependencies: Mapping[str, Tuple[str, ...]]) -> Tuple[str, ...]:
        state: Dict[str, int] = {}
        result = []

        def visit(device_id: str) -> None:
            marker = state.get(device_id, 0)
            if marker == 1:
                raise DigitalJSCircuitFormatError("DigitalJS connectors contain a cycle")
            if marker == 2:
                return
            state[device_id] = 1
            for dependency in dependencies[device_id]:
                visit(dependency)
            state[device_id] = 2
            result.append(device_id)

        for device_id in sorted(dependencies):
            visit(device_id)
        return tuple(result)

    @property
    def input_orders(self) -> Tuple[int, ...]:
        return tuple(order for order, _ in self._input_pairs)

    @property
    def output_orders(self) -> Tuple[int, ...]:
        return tuple(order for order, _ in self._output_pairs)

    @property
    def required_input_width(self) -> int:
        return 0 if not self._input_pairs else self._input_pairs[-1][0] + 1

    @property
    def output_count(self) -> int:
        return len(self._output_pairs)

    @property
    def topological_device_ids(self) -> Tuple[str, ...]:
        return tuple(device.device_id for device in self._compiled)

    @property
    def device_count(self) -> int:
        return len(self._compiled)

    @property
    def device_type_counts(self) -> Dict[str, int]:
        """Return a defensive copy of canonical device counts."""

        return dict(self._device_type_count_items)

    @property
    def logic_node_count(self) -> int:
        """Count logic/constant devices, excluding external Button/Output IO."""

        return self.device_count - len(self._input_pairs) - len(self._output_pairs)

    @property
    def critical_depth(self) -> int:
        """Maximum number of logic nodes on any path feeding an output."""

        return self._critical_depth

    @property
    def raw_connector_count(self) -> int:
        return self._raw_connector_count

    @property
    def unique_connector_count(self) -> int:
        return self._unique_connector_count

    def stats(self) -> Dict[str, Any]:
        """Return public, serialization-friendly circuit structure statistics."""

        return {
            "device_count": self.device_count,
            "device_type_counts": self.device_type_counts,
            "input_count": len(self._input_pairs),
            "output_count": len(self._output_pairs),
            "logic_node_count": self.logic_node_count,
            "critical_depth": self.critical_depth,
            "raw_connector_count": self.raw_connector_count,
            "unique_connector_count": self.unique_connector_count,
        }

    def evaluate(self, ordered_inputs: Any) -> np.ndarray:
        """Evaluate batches whose final axis is addressed by ``Button.order``.

        The final input axis may be wider than ``required_input_width``.  This
        matters for pruned artifacts: unused 3x3xC patch entries have no Button
        device, while the retained Buttons keep their original absolute order.
        Outputs are compacted in ascending ``Output.order``.
        """

        inputs = _binary_array(ordered_inputs, "ordered_inputs")
        if inputs.ndim == 0:
            raise ValueError("ordered_inputs needs a final feature axis")
        if inputs.shape[-1] < self.required_input_width:
            raise ValueError(
                "ordered_inputs has width %d but circuit order requires at least %d"
                % (inputs.shape[-1], self.required_input_width)
            )
        batch_shape = inputs.shape[:-1]
        values: Dict[str, np.ndarray] = {}
        for device in self._compiled:
            device_type = device.device_type
            if device_type == "Button":
                values[device.device_id] = inputs[..., self._input_order_by_id[device.device_id]]
            elif device_type == "False":
                values[device.device_id] = np.zeros(batch_shape, dtype=np.bool_)
            elif device_type == "True":
                values[device.device_id] = np.ones(batch_shape, dtype=np.bool_)
            elif device_type == "Output":
                values[device.device_id] = values[device.sources[0]]
            else:
                values[device.device_id] = _evaluate_gate(
                    device_type,
                    tuple(values[source] for source in device.sources),
                )
        if not self._output_pairs:
            return np.empty(batch_shape + (0,), dtype=np.uint8)
        outputs = np.stack(
            [values[device_id] for _, device_id in self._output_pairs],
            axis=-1,
        )
        return outputs.astype(np.uint8, copy=False)


def parse_digitaljs_json(document: Union[Mapping[str, Any], str, bytes]) -> DigitalJSBooleanCircuit:
    """Parse a mapping, UTF-8 JSON string, or UTF-8 JSON bytes."""

    if isinstance(document, Mapping):
        decoded = document
    elif isinstance(document, bytes):
        decoded = _validate_json_object(document)
    elif isinstance(document, str):
        try:
            decoded = json.loads(document)
        except json.JSONDecodeError as exc:
            raise DigitalJSCircuitFormatError("invalid JSON text") from exc
        if not isinstance(decoded, Mapping):
            raise DigitalJSCircuitFormatError("the JSON root must be an object")
    else:
        raise TypeError("document must be a mapping, JSON string, or JSON bytes")
    return DigitalJSBooleanCircuit(decoded)


def load_digitaljs_json(path: PathLike) -> DigitalJSBooleanCircuit:
    """Load and compile a DigitalJS JSON file from disk."""

    return parse_digitaljs_json(Path(path).read_bytes())


def flatten_3x3_patches(patches: Any, *, layout: str = "spatial_major") -> np.ndarray:
    """Flatten ``[..., 3, 3, C]`` with an explicit DigitalJS input layout.

    ``spatial_major`` uses ``order = f*C+c`` and remains the default.
    ``channel_major`` uses ``order = c*9+f``, matching the Button orders in
    the published multi-channel checkerboard artifact.
    """

    array = np.asarray(patches)
    if array.ndim < 3 or array.shape[-3] != 3 or array.shape[-2] != 3 or array.shape[-1] == 0:
        raise ValueError("patches must end in nonempty shape (3, 3, C)")
    channels = int(array.shape[-1])
    spatial_channel = array.reshape(array.shape[:-3] + (9, channels))
    if layout == "spatial_major":
        return spatial_channel.reshape(array.shape[:-3] + (9 * channels,))
    if layout == "channel_major":
        return np.swapaxes(spatial_channel, -2, -1).reshape(
            array.shape[:-3] + (9 * channels,)
        )
    raise ValueError("layout must be 'spatial_major' or 'channel_major'")


def _extract_3x3_patches(
    grid: np.ndarray,
    boundary: str,
    boundary_value: int,
) -> np.ndarray:
    if boundary == "periodic":
        padded = np.pad(grid, ((1, 1), (1, 1), (0, 0)), mode="wrap")
    elif boundary == "fixed":
        padded = np.pad(
            grid,
            ((1, 1), (1, 1), (0, 0)),
            mode="constant",
            constant_values=boundary_value,
        )
    else:
        raise ValueError("boundary must be 'periodic' or 'fixed'")
    height, width, channels = grid.shape
    patches = np.empty((height, width, 3, 3, channels), dtype=np.bool_)
    for row in range(3):
        for column in range(3):
            patches[:, :, row, column, :] = padded[
                row : row + height,
                column : column + width,
                :,
            ]
    return patches


def synchronous_ca_step(
    circuit: DigitalJSBooleanCircuit,
    grid: Any,
    *,
    boundary: str = "fixed",
    boundary_value: int = 0,
    input_layout: str = "spatial_major",
) -> np.ndarray:
    """Apply one synchronous DigitalJS 3x3xC cellular-automaton step."""

    if not isinstance(circuit, DigitalJSBooleanCircuit):
        raise TypeError("circuit must be a DigitalJSBooleanCircuit")
    state = _binary_array(grid, "grid")
    if state.ndim != 3 or min(state.shape) == 0:
        raise ValueError("grid must have nonempty shape (height, width, channels)")
    boundary_bit = _binary_array([boundary_value], "boundary_value")[0]
    patches = _extract_3x3_patches(state, boundary, int(boundary_bit))
    flattened = flatten_3x3_patches(patches, layout=input_layout)
    channels = int(state.shape[-1])
    expected_output_orders = tuple(range(channels))
    if circuit.output_orders != expected_output_orders:
        raise ValueError(
            "CA output orders must be contiguous channels 0..C-1; received %r"
            % (circuit.output_orders,)
        )
    return circuit.evaluate(flattened)


def synchronous_ca_rollout(
    circuit: DigitalJSBooleanCircuit,
    initial_grid: Any,
    steps: int,
    *,
    boundary: str = "fixed",
    boundary_value: int = 0,
    input_layout: str = "spatial_major",
) -> np.ndarray:
    """Return synchronous states from ``t=0`` through ``t=steps``."""

    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)):
        raise TypeError("steps must be an integer")
    steps = int(steps)
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    state = _binary_array(initial_grid, "initial_grid")
    if state.ndim != 3 or min(state.shape) == 0:
        raise ValueError("initial_grid must have nonempty shape (height, width, channels)")
    trajectory = np.empty((steps + 1,) + state.shape, dtype=np.uint8)
    trajectory[0] = state
    for step in range(steps):
        state = synchronous_ca_step(
            circuit,
            state,
            boundary=boundary,
            boundary_value=boundary_value,
            input_layout=input_layout,
        ).astype(np.bool_, copy=False)
        trajectory[step + 1] = state
    return trajectory
