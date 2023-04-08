`ocaml-gccjit` is a OCaml library that provides bidings for
[`libgccjit`](https://gcc.gnu.org/wiki/JIT).  `libgccjit` is an embeddable
shared library included in GCC 5 for adding compilation to existing programs
using GCC as the backend.

For example, consider this C function:

```c
int square (int i)
{
  return i * i;
}
```

We can construct this function at runtime using `libgccjit`, as follows:

```ocaml
open Gccjit

let square =
  let ctx = Context.create () in

  (* Create parameter "i" *)
  let param_i = Param.create ctx Type.(get ctx Int) "i" in

  (* Create the function *)
  let fn = Function.create ctx Function.Exported Type.(get ctx Int) "square" [ param_i ] in

  (* Create a basic block within the function *)
  let block = Block.create ~name:"entry" fn in

  (* This basic block is relatively simple *)
  let expr = RValue.binary_op ctx Mult Type.(get ctx Int) (RValue.param param_i) (RValue.param param_i) in
  Block.return block expr;

  (* Having populated the context, compile it *)
  let jit_result = Context.compile ctx in

  (* Look up a specific machine code routine within the gccjit.Result, in this
     case, the function we created above: *)
  Result.code jit_result "square" Ctypes.(int @-> returning int)
```

We can now call the function by doing simply
```ocaml
(* Now try running the code *)
Printf.printf "square(5) = %d\n%!" (square 5)
```

### Installation

Make sure that you installed the version of `libgccjit-VER-dev` corresponding to what `$ gcc --version` returns. Make sure that you can run the "hello world" example from the [libgccjit wiki](https://gcc.gnu.org/wiki/JIT). If `dune build` still fails, debug the build errors.


### Links

- [API documentation](https://nojb.github.io/ocaml-gccjit)
- [Tutorial](https://github.com/nojb/ocaml-gccjit/wiki)
- [The C header file](https://github.com/gcc-mirror/gcc/blob/master/gcc/jit/libgccjit.h)
- [libgccjit wiki](https://gcc.gnu.org/wiki/JIT)
- [Experiments in JIT compilation](https://github.com/davidmalcolm/jittest)

### Contact

Nicolas Ojeda Bar: <n.oje.bar@gmail.com>
