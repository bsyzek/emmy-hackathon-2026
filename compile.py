"""Compile an Emmy Kernel IR file to PTX.

Usage: python compile.py path/to/kernel.ir.json > out.ptx

The compiler reads the IR graph (plus the meta.json next to it, for the parameter order and
buffer shapes), finds the single KernelOp, and translates its Tile body into PTX:

1. Header and signature: .version/.target, then one .u64 pointer parameter per buffer.
2. Prologue: load each pointer, compute the thread's global index, guard threads past the
   Tile's last cell, and split the index into the Tile's named axes.
3. Body: walk the IR statements in order, one handler per statement type. A KernelBuilder
   hands out fresh virtual registers and maps IR value names to the registers holding them.
4. Output: the register declarations are printed first, though they are only known last.
"""
import json
import math
import struct
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import KernelOp, Smem, Sync, TreeHalve, WarpShuffle
from emmy.compiler.ir.stmt.blocks import Cond, StridedLoop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Let, Load, Write


def ptx_f32(x: float) -> str:
    """Return x as an exact PTX f32 literal: 0f + its 8 hex IEEE-754 digits, e.g. 1.0 -> 0f3F800000."""
    return "0f" + struct.pack(">f", x).hex().upper()


# --- Translation tables ---------------------------------------------------------------------

LOG2_E = math.log2(math.e)

EXIT_LABEL = "DONE"

# Register family prefix -> PTX register type.
PTX_REG_TYPES = {
    "f": ".f32",
    "rd": ".u64",
    "r": ".u32",
    "p": ".pred",
}

# Integer operators in index expressions -> PTX instruction.
INDEX_OPS = {
    "+": "add.u32",
    "*": "mul.lo.u32",
}

# Float elementwise op name -> (PTX instruction, extra operands appended after the arguments).
# exp is not here: it takes two instructions (see KernelBuilder.emit_assign).
FLOAT_OPS = {
    "add": ("add.f32", []),
    "subtract": ("sub.f32", []),
    "multiply": ("mul.f32", []),
    "divide": ("div.rn.f32", []),
    "reciprocal": ("rcp.rn.f32", []),
    "sqrt": ("sqrt.rn.f32", []),
    "maximum": ("max.f32", []),
    "relu": ("max.f32", [ptx_f32(0.0)]),
    "copy": ("mov.f32", []),
}

# Comparison operators in Cond -> setp comparison.
COMPARISON_TABLE = {
    "==": "eq",
    "!=": "ne",
    "<": "lt",
    "<=": "le",
    ">": "gt",
    ">=": "ge",
}


def float_op(name: str) -> tuple[str, list[str]]:
    """Return (instruction, extra operands) for a float elementwise op; raise if unsupported."""
    if name not in FLOAT_OPS:
        raise NotImplementedError(f"float op {name!r} not supported")
    return FLOAT_OPS[name]


class KernelBuilder:
    """Accumulates a kernel body's PTX lines and hands out fresh virtual registers."""

    # --- State and helpers ------------------------------------------------------------------

    def __init__(self):
        self.lines = []
        self.smem_decls = []
        self.shared = set()
        self.buffers = {}
        self.values = {}
        self.labels = {}
        self.shapes = {}
        self.global_idx = None
        self.reg_counts = {
            "f": 0,
            "rd": 0,
            "r": 0,
            "p": 0,
        }

    def set_shapes(self, args):
        """Record each global buffer's shape from meta.json's args, for flattening indices."""
        self.shapes = {a["name"]: a["shape"] for a in args}

    def new_reg(self, kind: str) -> str:
        """Return the next unused register name in a family, e.g. new_reg("rd") -> "%rd3"."""
        n = self.reg_counts[kind]
        name = f"%{kind}{n}"
        self.reg_counts[kind] += 1
        return name

    def new_label(self, prefix: str) -> str:
        """Return a fresh, unique label name like LOOP_0; emits nothing."""
        n = self.labels.get(prefix, 0)
        self.labels[prefix] = n + 1
        return f"{prefix}_{n}"

    def emit(self, line: str):
        """Append one indented instruction line to the body."""
        self.lines.append(f"    {line}")

    def emit_label(self, label: str):
        """Append a label line (unindented, no semicolon) as a branch target."""
        self.lines.append(f"{label}:")

    def reg_decls(self) -> list[str]:
        """Return one .reg declaration per register family in use, sized by its count."""
        reg_decl_lines = []
        for reg_prefix, count in self.reg_counts.items():
            if count != 0:
                line = f"    .reg {PTX_REG_TYPES[reg_prefix]} %{reg_prefix}<{count}>;"
                reg_decl_lines.append(line)

        return reg_decl_lines

    # --- Prologue ---------------------------------------------------------------------------

    def emit_param_loads(self, arg_order: list[str]):
        """Load each buffer's pointer param and convert it to a global address; record it in self.buffers."""
        for arg in arg_order:
            param_reg = self.new_reg("rd")
            self.emit(f"ld.param.u64 {param_reg}, [param_{arg}];")

            global_reg = self.new_reg("rd")
            self.emit(f"cvta.to.global.u64 {global_reg}, {param_reg};")

            self.buffers[arg] = global_reg

    def emit_thread_index(self):
        """Compute this thread's global index (ctaid.x * ntid.x + tid.x) into self.global_idx, plus lane and warp."""
        block_idx = self.new_reg("r")
        self.emit(f"mov.u32 {block_idx}, %ctaid.x;")

        block_size = self.new_reg("r")
        self.emit(f"mov.u32 {block_size}, %ntid.x;")

        thread_idx = self.new_reg("r")
        self.emit(f"mov.u32 {thread_idx}, %tid.x;")

        global_idx = self.new_reg("r")
        self.emit(f"mad.lo.u32 {global_idx}, {block_idx}, {block_size}, {thread_idx};")

        lane = self.new_reg("r")
        self.emit(f"and.b32 {lane}, {thread_idx}, 31;")

        warp = self.new_reg("r")
        self.emit(f"shr.u32 {warp}, {thread_idx}, 5;")

        self.global_idx = global_idx
        self.values["lane"] = lane
        self.values["warp"] = warp

    def emit_bounds_guard(self, total_cells: int):
        """Send threads past the Tile's last cell (global index >= total_cells) straight to the exit."""
        pred = self.new_reg("p")
        self.emit(f"setp.ge.u32 {pred}, {self.global_idx}, {total_cells};")
        self.emit(f"@{pred} bra {EXIT_LABEL};")

    def emit_tile_axes(self, axes):
        """Split the global index into one value per Tile axis, last axis fastest: (gid / stride) % extent."""
        for i, axis in enumerate(axes):
            extent = axis.extent.as_static()
            stride = math.prod(a.extent.as_static() for a in axes[i + 1:])

            value = self.global_idx
            if stride > 1:
                r = self.new_reg("r")
                self.emit(f"div.u32 {r}, {value}, {stride};")
                value = r

            if i > 0:
                r = self.new_reg("r")
                self.emit(f"rem.u32 {r}, {value}, {extent};")
                value = r

            self.values[axis.name] = value

    def emit_exit(self):
        """End the body with the exit label (the bounds guard's target) and ret."""
        self.emit_label(EXIT_LABEL)
        self.emit("ret;")

    # --- Expressions and addresses ----------------------------------------------------------

    def emit_expr(self, expr) -> str:
        """Emit PTX for an integer index expression tree; return the %r register holding its value."""
        if isinstance(expr, Var):
            return self.values[expr.name]
        elif isinstance(expr, Literal):
            reg_name = self.new_reg("r")
            self.emit(f"mov.u32 {reg_name}, {expr.value};")
            return reg_name
        elif isinstance(expr, BinaryExpr):
            left = self.emit_expr(expr.left)
            right = self.emit_expr(expr.right)

            if expr.op not in INDEX_OPS:
                raise NotImplementedError(f"index op {expr.op!r} not supported")

            instruction = INDEX_OPS[expr.op]
            dst = self.new_reg("r")
            self.emit(f"{instruction} {dst}, {left}, {right};")
            return dst
        else:
            raise NotImplementedError(f"index expression type {type(expr).__name__} not supported")

    def emit_address(self, buffer: str, index, nbytes: int) -> str:
        """Emit the address of buffer[index] (global or shared) for nbytes-wide elements; return its %rd register.

        An empty index is the buffer's first element, so the base address is returned as-is.
        A multi-dimensional index is flattened row-major using the buffer's shape.
        """
        if len(index) == 0:
            return self.buffers[buffer]

        dims = self.shapes[buffer]
        if len(dims) != len(index):
            raise ValueError(f"{buffer}: {len(index)}-D index into a {len(dims)}-D buffer {dims}")

        # Row-major flattening (Horner): flat = (...(i0 * d1 + i1) * d2 + i2 ...)
        flat = self.emit_expr(index[0])
        for k in range(1, len(index)):
            idx = self.emit_expr(index[k])
            new_flat = self.new_reg("r")
            self.emit(f"mad.lo.u32 {new_flat}, {flat}, {dims[k]}, {idx};")
            flat = new_flat

        byte_offset = self.new_reg("rd")
        self.emit(f"mul.wide.u32 {byte_offset}, {flat}, {nbytes};")

        offset_addr = self.new_reg("rd")
        self.emit(f"add.u64 {offset_addr}, {self.buffers[buffer]}, {byte_offset};")

        return offset_addr

    # --- Statements -------------------------------------------------------------------------

    def emit_body(self, stmts):
        """Translate each body statement in order, dispatching on its type."""
        for stmt in stmts:
            if isinstance(stmt, Load):
                self.emit_load(stmt)
            elif isinstance(stmt, Assign):
                self.emit_assign(stmt)
            elif isinstance(stmt, Write):
                self.emit_write(stmt)
            elif isinstance(stmt, Let):
                self.emit_let(stmt)
            elif isinstance(stmt, Accum):
                self.emit_accum(stmt)
            elif isinstance(stmt, StridedLoop):
                self.emit_strided_loop(stmt)
            elif isinstance(stmt, Cond):
                self.emit_cond(stmt)
            elif isinstance(stmt, Smem):
                self.emit_smem(stmt)
            elif isinstance(stmt, Sync):
                self.emit_sync(stmt)
            elif isinstance(stmt, WarpShuffle):
                self.emit_warp_shuffle(stmt)
            elif isinstance(stmt, TreeHalve):
                self.emit_tree_halve(stmt)
            else:
                raise NotImplementedError(f"stmt type {type(stmt).__name__} not supported")

    def emit_load(self, stmt):
        """Emit ld.global/ld.shared.f32 for a scalar Load into a fresh %f; record it under the Load's name."""
        if len(stmt.names) != 1:
            raise NotImplementedError(f"vectorized load of {len(stmt.names)} values not supported")

        addr = self.emit_address(stmt.input, stmt.index, stmt.dtype.nbytes)
        value = self.new_reg("f")
        space = "shared" if stmt.input in self.shared else "global"
        self.emit(f"ld.{space}.f32 {value}, [{addr}];")
        self.values[stmt.names[0]] = value

    def emit_assign(self, stmt):
        """Emit one float elementwise op over already-computed values; record the result under its name."""
        if stmt.op.name == "exp":
            # PTX only has 2^x (ex2), so e^x = 2^(x * log2(e)).
            x = self.values[stmt.args[0]]
            t = self.new_reg("f")
            self.emit(f"mul.f32 {t}, {x}, {ptx_f32(LOG2_E)};")

            dst = self.new_reg("f")
            self.emit(f"ex2.approx.f32 {dst}, {t};")

            self.values[stmt.name] = dst
            return

        instr, extras = float_op(stmt.op.name)
        registers = [self.values[a] for a in stmt.args]
        operands = registers + extras

        dst = self.new_reg("f")
        self.emit(f"{instr} {dst}, {", ".join(operands)};")
        self.values[stmt.name] = dst

    def emit_write(self, stmt):
        """Emit a store of 1, 2 or 4 consecutive f32 values (st / .v2 / .v4, global or shared) to output[index]."""
        if stmt.atomic or stmt.swizzle != "NONE":
            raise NotImplementedError(
                f"Write to {stmt.output}: atomic={stmt.atomic}, swizzle={stmt.swizzle} not supported")

        count = len(stmt.values)
        if count not in (1, 2, 4):
            raise NotImplementedError(
                f"Write to {stmt.output}: {count} values not supported (expected 1, 2 or 4)")

        addr = self.emit_address(stmt.output, stmt.index, stmt.value_dtype.nbytes)
        registers = [self.values[a] for a in stmt.values]
        space = "shared" if stmt.output in self.shared else "global"

        if count == 1:
            self.emit(f"st.{space}.f32 [{addr}], {registers[0]};")
        else:
            operands = ", ".join(registers)
            self.emit(f"st.{space}.v{count}.f32 [{addr}], {{{operands}}};")

    def emit_let(self, stmt):
        """Bind a name to a float constant (e.g. one = 1.0) in a fresh %f register."""
        if not isinstance(stmt.value, Literal):
            raise NotImplementedError(f"Let of {type(stmt.value).__name__} not supported")

        f = self.new_reg("f")
        self.emit(f"mov.f32 {f}, {ptx_f32(stmt.value.value)};")
        self.values[stmt.name] = f

    def emit_accum(self, stmt):
        """Fold a value into its loop accumulator in place; no new register.

        Normally acc = op(acc, value). With a base (online softmax rescales the old sum first),
        acc = op(base, value): the base replaces the old accumulator as the first operand.
        """
        accum = self.values[stmt.name]
        val = self.values[stmt.value]
        first = accum if stmt.base is None else self.values[stmt.base]

        instr, _ = float_op(stmt.op.name)
        self.emit(f"{instr} {accum}, {first}, {val};")

    def emit_strided_loop(self, stmt):
        """Emit a loop from start to the axis extent in steps of step, seeding its accumulators first."""
        for s in stmt.body:
            if isinstance(s, Accum):
                reg = self.new_reg("f")
                self.emit(f"mov.f32 {reg}, {ptx_f32(s.op.identity)};")
                self.values[s.name] = reg

        start = self.emit_expr(stmt.start)
        step = self.emit_expr(stmt.step)
        counter = self.new_reg("r")
        self.emit(f"mov.u32 {counter}, {start};")
        self.values[stmt.axis.name] = counter

        top = self.new_label("LOOP")
        end = f"{top}_END"

        self.emit_label(top)

        pred = self.new_reg("p")
        self.emit(f"setp.ge.u32 {pred}, {counter}, {stmt.axis.extent.as_static()};")
        self.emit(f"@{pred} bra {end};")

        self.emit_body(stmt.body)

        self.emit(f"add.u32 {counter}, {counter}, {step};")
        self.emit(f"bra {top};")
        self.emit_label(end)

    def emit_cond(self, stmt):
        """Run the body only where the comparison holds: setp, then branch past the body when false."""
        if not isinstance(stmt.cond, BinaryExpr) or stmt.cond.op not in COMPARISON_TABLE:
            raise NotImplementedError(
                f"Cond with {type(stmt.cond).__name__} {getattr(stmt.cond, 'op', '')!r} not supported")

        if stmt.else_body:
            raise NotImplementedError("Cond with an else_body not supported")

        cmp = COMPARISON_TABLE[stmt.cond.op]
        left = self.emit_expr(stmt.cond.left)
        right = self.emit_expr(stmt.cond.right)

        pred = self.new_reg("p")
        self.emit(f"setp.{cmp}.u32 {pred}, {left}, {right};")

        skip = self.new_label("SKIP")
        self.emit(f"@!{pred} bra {skip};")

        self.emit_body(stmt.body)
        self.emit_label(skip)

    def emit_smem(self, stmt):
        """Declare a module-level .shared array and record its base address as a shared buffer."""
        if stmt.dtype != "float":
            raise NotImplementedError(f"Smem of {stmt.dtype} not supported")

        name = stmt.name
        align = stmt.align or 4
        size = math.prod(stmt.extents) * 4
        self.shapes[name] = list(stmt.extents)

        self.smem_decls.append(f".shared .align {align} .b8 {name}[{size}];")

        base_addr = self.new_reg("rd")
        self.emit(f"mov.u64 {base_addr}, {name};")

        self.buffers[name] = base_addr
        self.shared.add(name)

    def emit_sync(self, stmt):
        """Emit a block-wide barrier (bar.sync): no thread continues until every thread arrives."""
        if stmt.warp or stmt.count is not None:
            raise NotImplementedError(f"Sync with warp={stmt.warp}, count={stmt.count} not supported")

        self.emit(f"bar.sync {stmt.barrier_id};")

    def emit_warp_shuffle(self, stmt):
        """Combine state across a warp with a shfl.sync.bfly butterfly (offsets length/2 .. 1)."""
        offset = stmt.length // 2
        while offset >= 1:
            for s, s_b in zip(stmt.state, stmt.state_b):
                src = self.values[s]
                r_in = self.new_reg("r")
                self.emit(f"mov.b32 {r_in}, {src};")
                r_out = self.new_reg("r")
                self.emit(f"shfl.sync.bfly.b32 {r_out}, {r_in}, {offset}, 0x1f, 0xffffffff;")
                f_other = self.new_reg("f")
                self.emit(f"mov.b32 {f_other}, {r_out};")
                self.values[s_b] = f_other

            for a in stmt.combine_states:
                self.emit_assign(a)

            offset //= 2

    def emit_tree_halve(self, stmt):
        """Combine the per-warp partials in shared memory; every thread ends up holding the total.

        Each thread seeds its state from slot 0, then folds in slots 1 .. length-1 with the
        IR's combine_states. All threads read the same addresses, which shared memory broadcasts.
        """
        if stmt.inner is not None:
            raise NotImplementedError("TreeHalve with an inner reduction not supported")

        for buf, s in zip(stmt.bufs, stmt.state):
            base = self.buffers[buf]
            f = self.new_reg("f")
            self.emit(f"ld.shared.f32 {f}, [{base}+0];")
            self.values[s] = f

        for i in range(1, stmt.length):
            for buf, s_b in zip(stmt.bufs, stmt.state_b):
                base = self.buffers[buf]
                f = self.new_reg("f")
                self.emit(f"ld.shared.f32 {f}, [{base}+{i * stmt.dtype.nbytes}];")
                self.values[s_b] = f

            for a in stmt.combine_states:
                self.emit_assign(a)


def emit_header() -> list[str]:
    """Return the fixed PTX module header lines (.version, .target, .address_size)."""
    header_lines = [
        ".version 8.0",
        ".target sm_89",
        ".address_size 64",
    ]

    return header_lines


def emit_signature(name: str, arg_order: list[str]) -> list[str]:
    """Return the .entry line and one .u64 pointer param per buffer, in arg_order."""
    signature_lines = []

    entry_start_line = f".visible .entry {name}("

    param_lines = []
    for i, param_name in enumerate(arg_order):
        line = f"    .param .u64 param_{param_name}"
        if i < len(arg_order) - 1:
            line += ","
        param_lines.append(line)

    entry_close_line = ")"

    signature_lines.append(entry_start_line)
    signature_lines.extend(param_lines)
    signature_lines.append(entry_close_line)

    return signature_lines


def get_kernel_op(g: Graph) -> KernelOp:
    """Return the graph's single KernelOp; raise if there isn't exactly one."""
    kernel_ops = []
    for node in g.nodes.values():
        if isinstance(node.op, KernelOp):
            kernel_ops.append(node.op)

    if len(kernel_ops) != 1:
        raise ValueError(f"expected 1 KernelOp, found {len(kernel_ops)}")

    return kernel_ops[0]


def get_kernel_meta(meta: dict, kernel_op_name: str) -> dict:
    """Return the meta.json kernel entry (arg_order, grid, block, ...) for this kernel."""
    kmeta = None
    for k in meta["kernels"]:
        if k["entry"] == kernel_op_name:
            kmeta = k

    if kmeta is None:
        raise ValueError(f"no entry for {kernel_op_name} in meta.json")

    return kmeta


def load_kernel(ir_path: Path) -> tuple[KernelOp, dict]:
    """Read the IR file and the meta.json beside it; return the KernelOp and its meta entry."""
    g = Graph.from_dict(json.loads(ir_path.read_text()))
    meta = json.loads((ir_path.parent / "meta.json").read_text())

    kernel_op = get_kernel_op(g)
    return kernel_op, get_kernel_meta(meta, kernel_op.name)


def compile_kernel(kernel_op: KernelOp, kernel_meta: dict) -> str:
    """Translate one KernelOp into a complete PTX module and return it as text."""
    kb = KernelBuilder()
    kb.set_shapes(kernel_meta["args"])

    # Prologue: buffer pointers, then which cell this thread runs.
    tile = kernel_op.body[0]
    kb.emit_param_loads(kernel_meta["arg_order"])
    kb.emit_thread_index()
    kb.emit_bounds_guard(math.prod(a.extent.as_static() for a in tile.axes))
    kb.emit_tile_axes(tile.axes)

    # Body: the IR statements, then the exit the bounds guard jumps to.
    kb.emit_body(tile.body)
    kb.emit_exit()

    # Assemble: register declarations go first, though they are only known now.
    ptx = [*emit_header(), ""]
    if kb.smem_decls:
        ptx += [*kb.smem_decls, ""]
    ptx += [
        *emit_signature(kernel_op.name, kernel_meta["arg_order"]),
        "{",
        *kb.reg_decls(),
        "",
        *kb.lines,
        "}",
    ]
    return "\n".join(ptx)


def main():
    """Read the IR path from argv, compile its kernel, and print the PTX to stdout."""
    if len(sys.argv) != 2:
        print("usage: python compile.py path/to/kernel.ir.json", file=sys.stderr)
        sys.exit(1)

    kernel_op, kernel_meta = load_kernel(Path(sys.argv[1]))
    print(compile_kernel(kernel_op, kernel_meta))


if __name__ == "__main__":
    main()
