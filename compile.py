"""Compile an Emmy Kernel IR file to PTX.

Usage: python compile.py path/to/kernel.ir.json > out.ptx
"""
import json
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel.ir import KernelOp
from emmy.compiler.ir.expr import Var, Literal, BinaryExpr

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
    "add": "add.f32",
    "multiply": "mul.f32"
}


class KernelBuilder:
    """Accumulates a kernel body's PTX lines and hands out fresh virtual registers."""

    def __init__(self):
        self.lines = []
        self.buffers = {}
        self.values = {}
        self.reg_counts = {
            "f": 0,
            "rd": 0,
            "r": 0,
            "p": 0 }

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

    def emit_thread_index(self, axis_name: str):
        """Compute this thread's global index (ctaid.x * ntid.x + tid.x); record it as axis_name."""
        block_idx = self.new_reg("r")
        self.emit(f"mov.u32 {block_idx}, %ctaid.x;")

        block_size = self.new_reg("r")
        self.emit(f"mov.u32 {block_size}, %ntid.x;")

        thread_idx = self.new_reg("r")
        self.emit(f"mov.u32 {thread_idx}, %tid.x;")

        global_idx = self.new_reg("r")
        self.emit(f"mad.lo.u32 {global_idx}, {block_idx}, {block_size}, {thread_idx};")

        self.values[axis_name] = global_idx

    def emit_bounds_guard(self, axis_name: str, extent: int):
        """Send threads whose index is outside the Tile (axis >= extent) straight to DONE."""
        global_idx = self.values[axis_name]

        pred = self.new_reg("p")
        self.emit(f"setp.ge.u32 {pred}, {global_idx}, {extent};")
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

    def emit_address(self, buffer: str, index_expr, nbytes: int) -> str:
        """Emit the global address of buffer[index_expr] for nbytes-wide elements; return its %rd register."""
        idx = self.emit_expr(index_expr)
        byte_offset = self.new_reg("rd")
        self.emit(f"mul.wide.u32 {byte_offset}, {idx}, {nbytes};")

        offset_addr = self.new_reg("rd")
        self.emit(f"add.u64 {offset_addr}, {self.buffers[buffer]}, {byte_offset};")

        return offset_addr

    def emit_load(self, stmt):
        """Emit ld.global.f32 for a scalar Load into a fresh %f; record it under the Load's name."""
        if len(stmt.names) != 1:
            raise NotImplementedError("vectorized loads aren't supported yet")

        addr = self.emit_address(stmt.input, stmt.index[0], stmt.dtype.nbytes)
        value = self.new_reg("f")
        self.emit(f"ld.global.f32 {value}, [{addr}];")
        self.values[stmt.names[0]] = value

    def emit_assign(self, stmt):
        """Emit one float elementwise op over already-computed values; record the result under its name."""
        if stmt.op.name not in FLOAT_OPS:
            raise NotImplementedError(f"float op {stmt.op.name} is not implemented yet")

        instr = FLOAT_OPS[stmt.op.name]
        registers = [self.values[a] for a in stmt.args]
        dst = self.new_reg("f")

        self.emit(f"{instr} {dst}, {", ".join(registers)};")
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

        addr = self.emit_address(stmt.output, stmt.index[0], stmt.value_dtype.nbytes)
        registers = [self.values[a] for a in stmt.values]

        if count == 1:
            self.emit(f"st.global.f32 [{addr}], {registers[0]};")
        else:
            operands = ", ".join(registers)
            self.emit(f"st.global.v{count}.f32 [{addr}], {{{operands}}};")

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
    kb.emit_param_loads(kernel_meta["arg_order"])

    tile = kernel_op.body[0]
    kb.emit_thread_index(tile.axes[0].name)
    kb.emit_bounds_guard(tile.axes[0].name,
                         tile.axes[0].extent.as_static())

    kb.emit_load(tile.body[0])
    kb.emit_load(tile.body[1])
    kb.emit_assign(tile.body[2])

    kb.emit_load(tile.body[3])
    kb.emit_load(tile.body[4])

    kb.emit_assign(tile.body[5])
    kb.emit_write(tile.body[6])

    print("\n".join(header))
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
