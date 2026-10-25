"""Compile an Emmy Kernel IR file to PTX.

Usage: python compile.py path/to/kernel.ir.json > out.ptx
"""
import json
import math
import struct
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel.ir import KernelOp, WarpShuffle, Smem, Sync, TreeHalve
from emmy.compiler.ir.expr import Var, Literal, BinaryExpr
from emmy.compiler.ir.stmt.leaves import Load, Assign, Write, Accum
from emmy.compiler.ir.stmt.blocks import StridedLoop, Cond

def ptx_f32(x: float) -> str:
    """Return x as an exact PTX f32 literal: 0f + its 8 hex IEEE-754 digits, e.g. 1.0 -> 0f3F800000."""
    return "0f" + struct.pack(">f", x).hex().upper()

PTX_REG_TYPES = {
    "f": ".f32",
    "rd": ".u64",
    "r": ".u32",
    "p": ".pred"
}

INDEX_OPS = {
    "+": "add.u32",
    "*": "mul.lo.u32"
}

FLOAT_OPS = {
    "add": ("add.f32", []),
    "multiply": ("mul.f32", []),
    "relu": ("max.f32", ["0f00000000"]),
    "divide": ("div.rn.f32", [])
}

COMPARISON_TABLE = {
    "==": "eq",
    "!=": "ne",
    "<": "lt",
    "<=": "le",
    ">": "gt",
    ">=": "ge"
}

class KernelBuilder:
    """Accumulates a kernel body's PTX lines and hands out fresh virtual registers."""

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
            "p": 0 }

    def set_shapes(self, args):
        self.shapes = {a["name"]: a["shape"] for a in args}

    def new_reg(self, kind: str) -> str:
        """Return the next unused register name in a family, e.g. new_reg("rd") -> "%rd3"."""
        n = self.reg_counts[kind]
        name = f"%{kind}{n}"
        self.reg_counts[kind] += 1
        return name

    def emit(self, line: str):
        """Append one indented instruction line to the body."""
        self.lines.append(f"    {line}")

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
        """Send threads past the Tile's last cell (global index >= total_cells) straight to DONE."""
        global_idx = self.global_idx

        pred = self.new_reg("p")
        self.emit(f"setp.ge.u32 {pred}, {global_idx}, {total_cells};")
        self.emit(f"@{pred} bra DONE;")

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
                raise NotImplementedError(f"index op {expr.op!r} is not supported")

            instruction = INDEX_OPS[expr.op]
            dst = self.new_reg("r")
            self.emit(f"{instruction} {dst}, {left}, {right};")
            return dst
        else:
            raise NotImplementedError(f"index expression type {type(expr).__name__}")

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

    def emit_load(self, stmt):
        """Emit ld.global.f32 for a scalar Load into a fresh %f; record it under the Load's name."""
        if len(stmt.names) != 1:
            raise NotImplementedError("vectorized loads aren't supported yet")

        addr = self.emit_address(stmt.input, stmt.index, stmt.dtype.nbytes)
        value = self.new_reg("f")
        space = "shared" if stmt.input in self.shared else "global"
        self.emit(f"ld.{space}.f32 {value}, [{addr}];")
        self.values[stmt.names[0]] = value

    def emit_assign(self, stmt):
        """Emit one float elementwise op over already-computed values; record the result under its name."""
        if stmt.op.name not in FLOAT_OPS:
            raise NotImplementedError(f"float op {stmt.op.name} is not implemented yet")

        instr, extras = FLOAT_OPS[stmt.op.name]
        registers = [self.values[a] for a in stmt.args]
        operands = registers + extras

        dst = self.new_reg("f")

        self.emit(f"{instr} {dst}, {", ".join(operands)};")
        self.values[stmt.name] = dst

    def emit_write(self, stmt):
        """Emit a store of 1, 2 or 4 consecutive f32 values (st.global / .v2 / .v4) to output[index]."""
        if stmt.atomic or stmt.swizzle != "NONE":
            raise NotImplementedError(
                f"Write to {stmt.output}: atomic={stmt.atomic}, swizzle={stmt.swizzle} not supported yet")

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

    def emit_body(self, stmts):
        """Translate each body statement in order, dispatching on its type."""
        for stmt in stmts:
            if isinstance(stmt, Load):
                self.emit_load(stmt)
            elif isinstance(stmt, Assign):
                self.emit_assign(stmt)
            elif isinstance(stmt, Write):
                self.emit_write(stmt)
            elif isinstance(stmt, Accum):
                self.emit_accum(stmt)
            elif isinstance(stmt, StridedLoop):
                self.emit_strided_loop(stmt)
            elif isinstance(stmt, WarpShuffle):
                self.emit_warp_shuffle(stmt)
            elif isinstance(stmt, Smem):
                self.emit_smem(stmt)
            elif isinstance(stmt, Cond):
                self.emit_cond(stmt)
            elif isinstance(stmt, Sync):
                self.emit_sync(stmt)
            elif isinstance(stmt, TreeHalve):
                self.emit_tree_halve(stmt)
            else:
                raise NotImplementedError(f"stmt type {type(stmt).__name__}")

    def emit_label(self, label: str):
        """Append a label line (unindented, no semicolon) as a branch target."""
        self.lines.append(f"{label}:")

    def emit_accum(self, stmt):
        """Fold a value into its loop accumulator in place (acc = op(acc, value)); no new register."""
        accum = self.values[stmt.name]
        val = self.values[stmt.value]

        if stmt.op.name not in FLOAT_OPS:
            raise NotImplementedError(f"float op {stmt.op.name} is not implemented yet")

        instr, extras = FLOAT_OPS[stmt.op.name]
        self.emit(f"{instr} {accum}, {accum}, {val};")

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

    def emit_smem(self, stmt):
        """Declare a module-level .shared array and record its base address as a shared buffer."""
        name = stmt.name
        align = stmt.align or 4
        size = math.prod(stmt.extents)
        self.shapes[name] = list(stmt.extents)

        if stmt.dtype != 'float':
            raise NotImplementedError(f"{stmt.dtype} not implemented yet")
        size *= 4

        self.smem_decls.append(f".shared .align {align} .b8 {name}[{size}];")

        base_addr = self.new_reg("rd")
        self.emit(f"mov.u64 {base_addr}, {name};")

        self.buffers[name] = base_addr
        self.shared.add(name)

    def emit_cond(self, stmt):
        """Run the body only where the comparison holds: setp, then branch past the body when false."""
        if (not isinstance(stmt.cond, BinaryExpr)) or (stmt.cond.op not in COMPARISON_TABLE):
            raise NotImplementedError(f"Cond with {type(stmt.cond).__name__} {getattr(stmt.cond, 'op', '')!r} not supported")

        if stmt.else_body:
            raise NotImplementedError("Cond with an else_body is not supported yet")

        cmp = COMPARISON_TABLE[stmt.cond.op]
        left = self.emit_expr(stmt.cond.left)
        right = self.emit_expr(stmt.cond.right)

        pred = self.new_reg("p")
        self.emit(f"setp.{cmp}.u32 {pred}, {left}, {right};")

        skip = self.new_label("SKIP")
        self.emit(f"@!{pred} bra {skip};")

        self.emit_body(stmt.body)
        self.emit_label(skip)

    def emit_sync(self, stmt):
        """Emit a block-wide barrier (bar.sync): no thread continues until every thread arrives."""
        if stmt.warp or (stmt.count is not None):
            raise NotImplementedError(f"Sync with warp={stmt.warp}, count={stmt.count} not supported yet")

        self.emit(f"bar.sync {stmt.barrier_id};")

    def emit_tree_halve(self, stmt):
        if stmt.inner is not None:
            raise NotImplementedError("TreeHalve with an inner reduction is not supported yet")

        for buf, s in zip(stmt.bufs, stmt.state):
            base = self.buffers[buf]
            f = self.new_reg("f")
            self.emit(f"ld.shared.f32 {f}, [{base}+0];")
            self.values[s] = f

        for i in range(1, stmt.length):
            for buf, s_b in zip(stmt.bufs, stmt.state_b):
                base = self.buffers[buf]
                f = self.new_reg("f")
                self.emit(f"ld.shared.f32 {f}, [{base}+{i*stmt.dtype.nbytes}];")
                self.values[s_b] = f

            for a in stmt.combine_states:
                self.emit_assign(a)

    def emit_tile_axes(self, axes):
        """Split the global index into one value per Tile axis, last axis fastest: (gid / stride) % extent."""
        for i, axis in enumerate(axes):
            extent = axis.extent.as_static()
            stride = math.prod(a.extent.as_static() for a in axes[i+1:])

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

    def new_label(self, prefix: str) -> str:
        """Return a fresh, unique label name like LOOP_0; emits nothing."""
        n = self.labels.get(prefix, 0)
        self.labels[prefix] = n + 1
        return f"{prefix}_{n}"

    def reg_decls(self) -> list[str]:
        """Return one .reg declaration per register family in use, sized by its count."""
        reg_decl_lines = []
        for reg_prefix, count in self.reg_counts.items():
            if count != 0:
                line = f"    .reg {PTX_REG_TYPES[reg_prefix]} %{reg_prefix}<{count}>;"
                reg_decl_lines.append(line)

        return reg_decl_lines


def emit_header() -> list[str]:
    """Return the fixed PTX module header lines (.version, .target, .address_size)."""
    header_lines = [
        ".version 8.0",
        ".target sm_89",
        ".address_size 64"
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


def main():
    if len(sys.argv) != 2:
        print("usage: python compile.py path/to/kernel.ir.json", file=sys.stderr)
        sys.exit(1)

    ir_path = Path(sys.argv[1])
    g = Graph.from_dict(json.load(open(ir_path)))
    meta = json.load(open(ir_path.parent / "meta.json"))

    kernel_op = get_kernel_op(g)
    kernel_meta = get_kernel_meta(meta, kernel_op.name)

    header = emit_header()
    signature = emit_signature(kernel_op.name, kernel_meta["arg_order"])

    kb = KernelBuilder()
    kb.set_shapes(kernel_meta["args"])
    kb.emit_param_loads(kernel_meta["arg_order"])

    tile = kernel_op.body[0]
    kb.emit_thread_index()
    kb.emit_bounds_guard(math.prod(a.extent.as_static() for a in tile.axes))
    kb.emit_tile_axes(tile.axes)
    kb.emit_body(tile.body)

    print("\n".join(header))
    print()
    if kb.smem_decls:
        print("\n".join(kb.smem_decls))
        print()
    print("\n".join(signature))
    print("{")
    print("\n".join(kb.reg_decls()))
    print()
    print("\n".join(kb.lines))
    print()
    print("DONE:")
    print("    ret;")
    print("}")


if __name__ == "__main__":
    main()
