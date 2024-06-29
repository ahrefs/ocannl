open Base

type 'context routine = {
  context : 'context;
  schedule : Tnode.task;
  bindings : Indexing.lowered_bindings;
  name : string;
}
[@@deriving sexp_of]

type config = Physical_devices_only | For_parallel_copying | Most_parallel_devices
[@@deriving equal, sexp, variants]

type merge_buffer_use = No | Streaming | Copy [@@deriving equal, sexp]

