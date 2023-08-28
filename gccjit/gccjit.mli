(* The MIT License (MIT)

   Copyright (c) 2015 Nicolas Ojeda Bar <n.oje.bar@gmail.com>

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE. *)

(** OCaml bindings for [libgccjit].

    See {{:https://gcc.gnu.org/wiki/JIT}GCC wiki page} for more information. *)

exception Error of string
(** This exception (containing an explanatory string) is raised if an error
    occurs. *)

type context
(** The type of compilation contexts.  See {{!contexts}Compilation
    Contexts}. *)

type result
(** A {!result} encapsulates the result of an in-memory compilation. *)

type location
(** A {!location} encapsulates a source code location, so that you can
    (optionally) associate locations in your languages with statements in the
    JIT-compiled code, alowing the debugger to single-step through your
    language. See {{!locations}Source locations}. *)

type param
(** A {!param} is a function parameter.  See {{!params}Parameters}. *)

type lvalue
(** An {!lvalue} is something that can be the left-hand side of an assignment.
    See {{!lvalues}Lvalues}. *)

type rvalue
(** A [rvalue] is an expression within your code, with some type. See
    {{!rvalues}RValues}. *)

type field
(** The type of fields of structs and unions.  See {{!fields}Fields}. *)

type struct_
(** The type of structure types. See {{!structs}Structure Types}. *)

type type_
(** The type of C types, e.g. [int] or a [struct foo*]. See {{!types}Types}. *)

type function_
(** The type of functios.  See {{!functions}Functions}. *)

type block
(** The type of basic blocks.  See {{!blocks}Basic Blocks}. *)

type unary_op = Negate | Bitwise_negate | Logical_negate

type binary_op =
  | Plus
  | Minus
  | Mult
  | Divide
  | Modulo
  | Bitwise_and
  | Bitwise_xor
  | Bitwise_or
  | Logical_and
  | Logical_or

type comparison = Eq | Ne | Lt | Le | Gt | Ge

(** {1:contexts Compilation Contexts}

    A {!context} encapsulates the state of a compilation.  You can
    {{!Context.set_option}set up options} on it, add {{!types}types},
    {{!functions}functions} and {{!blocks}code}, using the API below.

    Invoking {!Context.compile} on it gives you a {!result}, representing
    in-memory machine-code.

    You can call {!Context.compile} repeatedly on one context, giving multiple
    independent results.

    Similarly, you can call {!Context.compile_to_file} on a context to compile
    to disk.

    Eventually you can call {!Context.release} to clean up the context; any
    in-memory results created from it are still usable. *)

val get_first_error : context -> string option

module Context : sig
  (** {1 Lifetime-management} *)

  val create : unit -> context
  (** Creates a new {!context} instance, which is independent of any others that
      may be present within this process. *)

  val release : context -> unit
  (** Releases all resources associated with the given context.  Both the
      context itself and all of its object instances are cleared up.  It should
      be called exactly once on a given context.

      It is invalid to use the context or any of its {e contextual} objects
      after calling this. *)

  val create_child : context -> context
  (** Given an existing JIT context, create a child context.
      - The child inherits a copy of all option-settings from the parent.
      - The child can reference objects created within the parent, but not
        vice-versa.
      - The lifetime of the child context must be bounded by that of the parent:
        you should release a child context before releasing the parent context.

      If you use a function from a parent context within a child context, you
      have to compile the parent context before you can compile the child
      context, and the {!result} of the parent context must outlive the
      {!result} of the child context.

      This allows caching of shared initializations. For example, you could
      create types and declarations of global functions in a parent context once
      within a process, and then create child contexts whenever a function or
      loop becomes hot. Each such child context can be used for JIT-compiling
      just one function or loop, but can reference types and helper functions
      created within the parent context.

      Contexts can be arbitrarily nested, provided the above rules are followed,
      but it's probably not worth going above 2 or 3 levels, and there will likely
      be a performance hit for such nesting. *)

  (** {1 Thread-safety}

      Instances of {!context} created via {!create} are independent from each
      other: only one thread may use a given context at once, but multiple
      threads could each have their own contexts without needing locks.

      Contexts created via {!create_child} are related to their parent
      context. They can be partitioned by their ultimate ancestor into
      independent "family trees". Only one thread within a process may use a
      given "family tree" of such contexts at once, and if you're using multiple
      threads you should provide your own locking around entire such context
      partitions. *)

  (** {1 Debugging} *)

  val dump_to_file : context -> ?update_locs:bool -> string -> unit
  (** Dump a C-like representation to the given path, describing what's been set
      up on the context.  If [~update_locs] true, then also set up {!location}
      information throughout the context, pointing at the dump file as if it
      were a source file.  This may be of use in conjunction with
      {{!Context.context_option}[Debuginfo]} to allow stepping through the code
      in a debugger. *)

  val set_logfile : context -> Unix.file_descr option -> unit
  (** [set_logfile ctx logfile] enable ongoing logging of the context [ctx]'s
      activity to the given file descriptor [logfile]. Examples of information
      logged include:
      - API calls
      - the various steps involved within compilation
      - activity on any {!result} instances created by the context
      - activity within any child contexts
      - An example of a log can be seen here, though the precise format and kinds of
        information logged is subject to change.

      The caller remains responsible for closing [logfile], and it must not be
      closed until all users are released. In particular, note that child
      {{!context}contexts} and {!result} instances created by the {!context}
      will use [logfile].

      There may a performance cost for logging.

      You can turn off logging on [ctx] by passing [None] for [logfile]. Doing
      so only affects the context; it does not affect child {{!context}contexts}
      or {!result} instances already created by the {!context}. *)

  val dump_reproducer_to_file : context -> string -> unit
  (** Write C source code into path that can be compiled into a self-contained
      executable (i.e. with [libgccjit] as the only dependency). The generated
      code will attempt to replay the API calls that have been made into the
      given context.

      This may be useful when debugging the library or client code, for reducing a
      complicated recipe for reproducing a bug into a simpler form. For example,
      consider client code that parses some source file into some internal
      representation, and then walks this IR, calling into [libgccjit]. If this
      encounters a bug, a call to {!dump_reproducer_to_file} will write out C code
      for a much simpler executable that performs the equivalent calls into
      [libgccjit], without needing the client code and its data.

      Typically you need to supply ["-Wno-unused-variable"] when compiling the
      generated file (since the result of each API call is assigned to a unique
      variable within the generated C source, and not all are necessarily then
      used). *)

  (** {1 Context Options} *)

  type _ context_option =
    | Progname : string context_option
        (** The name of the program, for used as a prefix when printing error messages
        to stderr.  If not set, ["libgccjit.so"] is used. *)
    | Optimization_level : int context_option
        (** How much to optimize the code.  Valid values are [0-3], corresponding to
        GCC's command-line options -O0 through -O3.

        The default value is 0 (unoptimized). *)
    | Debuginfo : bool context_option
        (** If [true], {!Context.compile} will attempt to do the right thing so that
        if you attach a debugger to the process, it will be able to inspect
        variables and step through your code.  Note that you can't step through
        code unless you set up source location information for the code (by
        creating and passing in {!location} instances).  *)
    | Dump_initial_tree : bool context_option
        (** If [true], {!Context.compile} will dump its initial "tree" representation
        of your code to [stderr] (before any optimizations).  *)
    | Dump_initial_gimple : bool context_option
        (** If [true], {!Context.compile} will dump the "gimple" representation of
        your code to stderr, before any optimizations are performed.  The dump
        resembles C code.  *)
    | Dump_generated_code : bool context_option
        (** If [true], {!Context.compile} will dump the final generated code to
        stderr, in the form of assembly language.  *)
    | Dump_summary : bool context_option
        (** If [true], {!Context.compile} will print information to stderr on the
        actions it is performing, followed by a profile showing the time taken and
        memory usage of each phase. *)
    | Dump_everything : bool context_option
        (** If [true], {!Context.compile} will dump copious amount of information on
        what it's doing to various files within a temporary directory.  Use
        [Keep_intermediates] (see below) to see the results.  The files are
        intended to be human-readable, but the exact files and their formats are
        subject to change. *)
    | Selfcheck_gc : bool context_option
        (** If [true], [libgccjit] will aggressively run its garbage collector,
        to shake out bugs (greatly slowing down the compile).  This is likely to
        only be of interest to developers *of* the library.  It is used when
        running the selftest suite.  *)
    | Keep_intermediates : bool context_option
        (** If [true], {!Context.release} will not clean up intermediate files written
         to the filesystem, and will display their location on stderr.  *)

  val set_option : context -> 'a context_option -> 'a -> unit
  (** [set_option ctx opt v] sets the {!Context.context_option} [opt] of [ctx]
      to [v]. *)

  (** {1 Compilation}

      Once populated, a {!context} can be compiled to machine code, either
      in-memory via {!compile} or to disk via {!compile_to_file}.

      You can compile a context multiple times (using either form of compilation),
      although any errors that occur on the context will prevent any future
      compilation of that context. *)

  val compile : context -> result
  (** This calls into GCC and builds the code, returning a {!result}.  See
      {{!inmemory}In-memory compilation}. *)

  (** Kinds of ahead-of-time compilation, for use with
      {!compile_to_file}.  *)
  type output_kind =
    | Assembler  (** Compile the context to an assembly file. *)
    | Object_file  (** Compile the context to an object file. *)
    | Dynamic_library  (** Compile the context to a dynamic library. *)
    | Executable  (** Compile the context to an executable. *)

  val compile_to_file : context -> output_kind -> string -> unit
  (** Compile the context to a file of the given kind.  This can be called more
      that once on a given context, although any errors that occur will block
      further compilation. *)
end

(** {1:fields Fields}

    A [field] encapsulates a field within a struct; it is used when creating a
    struct type (using {!Struct.create}).  Fields cannot be shared between
    structs. *)

module Field : sig
  val create : context -> ?loc:location -> type_ -> string -> field
  (** Create a field, with the given type and name. *)

  val to_string : field -> string
  (** Get a human-readable description of this object. *)
end

(** {1:structs Structure Types}

    A [struct_] encapsulates a struct type, either one that we have the layout for,
    or an opaque type.

    You can model C struct types by creating [struct_] and [field] instances, in
    either order:

    - by creating the fields, then the structure. For example, to model:
{[
struct coord {double x; double y; };
]}
      you could call:
{[
let field_x = Field.create ctx double_type "x" in
let field_y = Field.create ctx double_type "y" in
let coord = Struct.create ctx "coord" [ field_x ; field_y ]
]}
    - by creating the structure, then populating it with fields, typically to
      allow modelling self-referential structs such as:
{[
struct node { int m_hash; struct node *m_next; };
]}
      like this:
{[
let node = Struct.create_opaque ctx "node" in
let node_ptr = Type.pointer node in
let field_hash = Field.create ctx int_type "m_hash" in
let field_next = Field.create ctx node_ptr "m_next" in
Struct.set_fields node [ field_hash; field_next ]
]} *)

module Struct : sig
  val create : context -> ?loc:location -> string -> field list -> struct_
  (** Create a struct type, with the given name and fields. *)

  val opaque : context -> ?loc:location -> string -> struct_
  (** Construct a new struct type, with the given name, but without specifying the
      fields. The fields can be omitted (in which case the size of the struct is not
      known), or later specified using {!set_fields}. *)

  val set_fields : ?loc:location -> struct_ -> field list -> unit
  (** Populate the fields of a formerly-opaque struct type.

      This can only be called once on a given struct type. *)

  val to_string : struct_ -> string
  (** Get a human-readable description of this object. *)
end

(** {1:types Types} *)

module Type : sig
  type type_kind =
    | Void
    | Void_ptr
    | Bool
    | Char
    | Signed_char
    | Unsigned_char
    | Short
    | Unsigned_short
    | Int
    | Unsigned_int
    | Long
    | Unsigned_long
    | Long_long
    | Unsigned_long_long
    | Float
    | Double
    | Long_double
    | Const_char_ptr
    | Size_t
    | File_ptr
    | Complex_float
    | Complex_double
    | Complex_long_double

  val get : context -> type_kind -> type_
  (** Access a standard type.  See {!type_kind}. *)

  val int : context -> ?signed:bool -> int -> type_
  (** Get the integer type of the given size and signedness. *)

  val pointer : type_ -> type_
  (** Given type [T], get type [T*] *)

  val const : type_ -> type_
  (** Given type [T], get type [const T]. *)

  val volatile : type_ -> type_
  (** Given type [T], get type [volatile T]. *)

  val array : context -> ?loc:location -> type_ -> int -> type_
  (** Given type [T], get type [T[N]] (for a constant [N]). *)

  val function_ptr : context -> ?loc:location -> ?variadic:bool -> type_ list -> type_ -> type_
  val struct_ : struct_ -> type_

  val union : context -> ?loc:location -> string -> field list -> type_
  (** Unions work similarly to structs. *)

  val to_string : type_ -> string
  (** Get a human-readable description of this object. *)
end

(** {1:rvalues Rvalues}

    A {!rvalue} is an expression that can be computed.

    It can be simple, e.g.:
    - an integer value e.g. [0] or [42]
    - a string literal e.g. ["Hello world"]
    - a variable e.g. [i]. These are also {{!lvalues}lvalues} (see below).

    or compound e.g.:
    - a unary expression e.g. [!cond]
    - a binary expression e.g. [(a + b)]
    - a function call e.g. [get_distance (&player_ship, &target)]
    - etc.

    Every {!rvalue} has an associated {{!type_}type}, and the API will check to
    ensure that types match up correctly (otherwise the context will emit an
    error). *)

module RValue : sig
  val type_of : rvalue -> type_
  (** Get the type of this {!rvalue}. *)

  val int : context -> type_ -> int -> rvalue
  (** Given a numeric type (integer or floating point), build an {!rvalue} for the
      given constant int value. *)

  val zero : context -> type_ -> rvalue
  (** Given a numeric type (integer or floating point), get the {!rvalue} for
      zero. Essentially this is just a shortcut for:
{[
new_rvalue_from_int ctx numeric_type 0
]} *)

  val one : context -> type_ -> rvalue
  (** Given a numeric type (integer or floating point), get the {!rvalue} for
      one. Essentially this is just a shortcut for:
{[
new_rvalue_from_int ctx numeric_type 1
]} *)

  val double : context -> type_ -> float -> rvalue
  (** Given a numeric type (integer or floating point), build an {!rvalue} for the
      given constant double value. *)

  val ptr : context -> type_ -> 'a Ctypes.ptr -> rvalue
  (** Given a pointer type, build an {!rvalue} for the given address. *)

  val null : context -> type_ -> rvalue
  (** Given a pointer type, build an {!rvalue} for [NULL]. Essentially this is
      just a shortcut for:
{[
new_rvalue_from_ptr ctx pointer_type Ctypes.null
]} *)

  val string_literal : context -> string -> rvalue
  (** Generate an {!rvalue} for the given [NIL]-terminated string, of type
      [Const_char_ptr]. *)

  val unary_op : context -> ?loc:location -> unary_op -> type_ -> rvalue -> rvalue
  (** Build a unary operation out of an input {!rvalue}.  See {!unary_op}. *)

  val binary_op : context -> ?loc:location -> binary_op -> type_ -> rvalue -> rvalue -> rvalue
  (** Build a binary operation out of two constituent {{!rvalue}rvalues}. See
      {!binary_op}. *)

  val comparison : context -> ?loc:location -> comparison -> rvalue -> rvalue -> rvalue
  (** Build a boolean {!rvalue} out of the comparison of two other
      {{!rvalue}rvalues}. *)

  val call : context -> ?loc:location -> function_ -> rvalue list -> rvalue
  (** Given a function and the given table of argument rvalues, construct a call
      to the function, with the result as an {!rvalue}.

      {3 Note}

      [new_call] merely builds a [rvalue] i.e. an expression that can be
      evaluated, perhaps as part of a more complicated expression.  The call won't
      happen unless you add a statement to a function that evaluates the expression.

      For example, if you want to call a function and discard the result (or to call a
      function with [void] return type), use [add_eval]:
{[
(* Add "(void)printf (args);". *)
add_eval block (new_call ctx printf_func args)
]} *)

  val indirect_call : context -> ?loc:location -> rvalue -> rvalue list -> rvalue
  (** Call through a function pointer. *)

  val cast : context -> ?loc:location -> rvalue -> type_ -> rvalue
  (** Given an {!rvalue} of [T], construct another {!rvalue} of another type.
      Currently only a limited set of conversions are possible:
      - [int <-> float]
      - [int <-> bool]
      - [P* <-> Q*], for pointer types [P] and [Q] *)

  val access_field : ?loc:location -> rvalue -> field -> rvalue
  val lvalue : lvalue -> rvalue
  val param : param -> rvalue

  val to_string : rvalue -> string
  (** Get a human-readable description of this object. *)
end

(** {1:lvalues Lvalues}

    An {!lvalue} is something that can of the left-hand side of an assignment: a
    storage area (such as a variable). It is also usable as an {!rvalue}, where
    the {!rvalue} is computed by reading from the storage area. *)

module LValue : sig
  val address : ?loc:location -> lvalue -> rvalue
  (** Taking the address of an {!lvalue}; analogous to [&(EXPR)] in C. *)

  type global_kind = Exported | Internal | Imported

  val global : context -> ?loc:location -> global_kind -> type_ -> string -> lvalue
  (** Add a new global variable of the given type and name to the context.

      The {!global_kind} parameter determines the visibility of the {e global}
      outside of the {!result}. *)

  val deref : ?loc:location -> rvalue -> lvalue
  (** Dereferencing a pointer; analogous to [*(EXPR)] in C. *)

  val deref_field : ?loc:location -> rvalue -> field -> lvalue
  (** Accessing a field of an [rvalue] of pointer type, analogous [(EXPR)->field]
      in C, itself equivalent to [(\*EXPR).FIELD] *)

  val access_array : ?loc:location -> rvalue -> rvalue -> lvalue
  (** Given an rvalue of pointer type [T *], get at the element [T] at the given
      index, using standard C array indexing rules i.e. each increment of index
      corresponds to [sizeof(T)] bytes. Analogous to [PTR[INDEX]] in C (or,
      indeed, to [PTR + INDEX]). *)

  val access_field : ?loc:location -> lvalue -> field -> lvalue
  val param : param -> lvalue

  val to_string : lvalue -> string
  (** Get a human-readable description of this object. *)
end

(** {1:params Parameters}

    A value of type {!param} represents a parameter to a
    {{!functions}function}. *)

module Param : sig
  val create : context -> ?loc:location -> type_ -> string -> param
  (** In preparation for creating a function, create a new parameter of the given
      type and name. *)

  val to_string : param -> string
  (** Get a human-readable description of this object. *)
end

(** {1:functions Functions}

    A values of type [function_] encapsulates a function: either one that you're
    creating yourself, or a reference to one that you're dynamically linking to
    within the rest of the process. *)

module Function : sig
  (** Kinds of function.  *)
  type function_kind =
    | Exported  (** Function is defined by the client code and visible by name outside of the
        JIT. *)
    | Internal
        (** Function is defined by the client code, but is invisible outside of the
        JIT.  Analogous to a ["static"] function. *)
    | Imported
        (** Function is not defined by the client code; we're merely referring to it.
         Analogous to using an ["extern"] function from a header file. *)
    | Always_inline
        (** Function is only ever inlined into other functions, and is invisible
        outside of the JIT.  Analogous to prefixing with ["inline"] and adding
        [__attribute__((always_inline))].  Inlining will only occur when the
        optimization level is above 0; when optimization is off, this is
        essentially the same as [FUNCTION_Internal]. *)

  val create :
    context -> ?loc:location -> ?variadic:bool -> function_kind -> type_ -> string -> param list -> function_
  (** Create a function with the given name and parameters. *)

  val builtin : context -> string -> function_
  (** Create a reference to a builtin function (sometimes called intrinsic
      functions). *)

  val param : function_ -> int -> param
  (** Get a specific param of a function by index (0-based). *)

  val dump_dot : function_ -> string -> unit
  (** Emit the function in graphviz format to the given path. *)

  val local : ?loc:location -> function_ -> type_ -> string -> lvalue
  (** Add a new local variable within the function, of the given type and name. *)

  val to_string : function_ -> string
  (** Get a human-readable description of this object. *)
end

(** {1:blocks Basic Blocks}

    A [block] encapsulates a {e basic block} of statements within a function
    (i.e. with one entry point and one exit point).
    - Every block within a function must be terminated with a conditional, a
      branch, or a return.
    - The blocks within a function form a directed graph.
    - The entrypoint to the function is the first block created within it.
    - All of the blocks in a function must be reachable via some path from the
      first block.
    - It's OK to have more than one {e return} from a function (i.e., multiple
      blocks that terminate by returning. *)

module Block : sig
  val create : ?name:string -> function_ -> block
  (** Create a block.  You can give it a meaningful name, which may show up in
      dumps of the internal representation, and in error messages. *)

  val parent : block -> function_
  (** Which function is this block within? *)

  val eval : ?loc:location -> block -> rvalue -> unit
  (** Add evaluation of an {!rvalue}, discarding the result (e.g. a function
      call that {e returns} void).  This is equivalent to this C code:
{[
(void)expression;
]} *)

  val assign : ?loc:location -> block -> lvalue -> rvalue -> unit
  (** Add evaluation of an {!rvalue}, assigning the result to the given
      {!lvalue}.  This is roughly equivalent to this C code:
{[
lvalue = rvalue;
]} *)

  val assign_op : ?loc:location -> block -> lvalue -> binary_op -> rvalue -> unit
  (** Add evaluation of an {!rvalue}, using the result to modify an {!lvalue}.
      This is analogous to ["+="] and friends:
{[
lvalue += rvalue;
lvalue *= rvalue;
lvalue /= rvalue;
etc
]}
      For example:
{[
(* "i++" *)
add_assignment_op loop_body i Plus (one ctx int_type)
]} *)

  val comment : ?loc:location -> block -> string -> unit
  (** Add a no-op textual comment to the internal representation of the code.
      It will be optimized away, but will be visible in the dumps seen via
      {{!Context.context_option}[Dump_initial_tree]} and
      {{!Context.context_option}[Dump_initial_gimple]} and thus may be of use
      when debugging how your project's internal representation gets converted
      to the [libgccjit] IR.  *)

  val cond_jump : ?loc:location -> block -> rvalue -> block -> block -> unit
  (** Terminate a block by adding evaluation of an rvalue, branching on the
      result to the appropriate successor block.  This is roughly equivalent to
      this C code:
{[
if (boolval)
  goto on_true;
else
  goto on_false;
]} *)

  val jump : ?loc:location -> block -> block -> unit
  (** Terminate a block by adding a jump to the given target block.  This is
      roughly equivalent to this C code:
{[
goto target;
]} *)

  val return : ?loc:location -> block -> rvalue -> unit
  (** Terminate a block by adding evaluation of an {!rvalue}, returning the
      value.  This is roughly equivalent to this C code:
{[
return expression;
]} *)

  val return_void : ?loc:location -> block -> unit
  (** Terminate a block by adding a valueless return, for use within a function
      with [void] return type.  This is equivalent to this C code:
{[
return;
]} *)

  val to_string : block -> string
  (** Get a human-readable description of this object. *)
end

(** {1:locations Source Locations}

    A {!location} encapsulates a source code location, so that you can (optionally)
    associate locations in your language with statements in the JIT-compiled code,
    allowing the debugger to single-step through your language.
    - {!location} instances are optional: you can always omit them to any API
      entrypoint accepting one.
    - You can construct them using {!Location.create}.
    - You need to {{!Context.set_option}enable}
      {{!Context.context_option}[Debuginfo]} on the {!context} for these
      locations to actually be usable by the debugger.

    {2 Faking it}

    If you don't have source code for your internal representation, but need to
    debug, you can generate a C-like representation of the functions in your
    context using {!Context.dump_to_file}.  This will dump C-like code to the
    given path. If the update_locations argument is true, this will also set up
    {!location} information throughout the context, pointing at the dump file as
    if it were a source file, giving you something you can step through in the
    debugger. *)

module Location : sig
  val create : context -> string -> int -> int -> location
  (** Create a {!location} instance representing the given source location. *)

  val to_string : location -> string
  (** Get a human-readable description of this object. *)
end

(** {1:inmemory In-memory compilation} *)

module Result : sig
  val code : result -> string -> ('a -> 'b) Ctypes.fn -> 'a -> 'b
  (** Locate a given function within the built machine code.
      - Functions are looked up by name. For this to succeed, a function with a
        name matching funcname must have been created on result's context (or a
        parent context) via a call to {!Function.create} with kind
        {{!Function.function_kind}[Exported]}.
      - If such a function is not found, an error will be raised.
      - If the function is found, the result is cast to the given Ctypes
        signature.  Care must be taken to pass a signature compatible with that
        of function being extracted.
      - The resulting machine code becomes invalid after {!release} is called on
        the {!result}; attempting to call it after that may lead to a
        segmentation fault. *)

  val global : result -> string -> 'a Ctypes.typ -> 'a Ctypes.ptr
  (** Locate a given global within the built machine code.
      - Globals are looked up by name. For this to succeed, a global with a name
        matching name must have been created on result's context (or a parent
        context) via a call to {!LValue.global} with kind
        {{!LValue.global_kind}[Exported]}.
      - If the global is found, the result is cast to the Given [Ctypes] type.
      - This is a pointer to the global, so e.g. for an [int] this is an [int *].
      - If such a global is not found, an error will be raised.
      - The resulting address becomes invalid after {!release} is called on the
        {!result}; attempting to use it after that may lead to a segmentation
        fault.  *)

  val release : result -> unit
  (** Once we're done with the code, this unloads the built [.so] file. This
      cleans up the result; after calling this, it's no longer valid to use the
      result, or any code or globals that were obtained by calling {!code} or
      {!global} on it. *)
end
