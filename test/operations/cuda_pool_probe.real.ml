open Base
module Cu = Cuda

type stats = {
  reserved : Unsigned.uint64;
  reserved_high : Unsigned.uint64;
  used : Unsigned.uint64;
  used_high : Unsigned.uint64;
}

type snapshot = {
  is_default : bool;
  threshold : Unsigned.uint64;
  stream : stats;
  graph : stats;
  free : int;
}

(* cudajit binds allocation/free and free-device-memory queries, but lacks both allocators'
   attributes. Keep this independent readout for checking the backend's reported gauges. *)
module Driver = struct
  open Ctypes

  let lib =
    lazy
      (let filename = if Stdlib.Sys.win32 then "nvcuda.dll" else "libcuda.so.1" in
       Dl.dlopen ~filename ~flags:[ Dl.RTLD_NOW ])

  let binding name signature = lazy (Foreign.foreign ~from:(Lazy.force lib) name signature)
  let current_device = binding "cuCtxGetDevice" (ptr int @-> returning int)
  let get_pool = binding "cuDeviceGetMemPool" (ptr (ptr void) @-> int @-> returning int)
  let get_default = binding "cuDeviceGetDefaultMemPool" (ptr (ptr void) @-> int @-> returning int)

  let pool_attribute =
    binding "cuMemPoolGetAttribute" (ptr void @-> int @-> ptr uint64_t @-> returning int)

  let graph_attribute =
    binding "cuDeviceGetGraphMemAttribute" (int @-> int @-> ptr uint64_t @-> returning int)

  let get name typ init call =
    let out = allocate typ init in
    let status = call out in
    if status <> 0 then failwith (Printf.sprintf "%s returned CUDA status %d" name status);
    !@out

  let device () = get "cuCtxGetDevice" int 0 (Lazy.force current_device)

  let pool device =
    get "cuDeviceGetMemPool" (ptr void) null (fun out -> (Lazy.force get_pool) out device)

  let default device =
    get "cuDeviceGetDefaultMemPool" (ptr void) null (fun out -> (Lazy.force get_default) out device)

  let bytes name query = get name uint64_t Unsigned.UInt64.zero query
  let pool_bytes pool attr = bytes "cuMemPoolGetAttribute" ((Lazy.force pool_attribute) pool attr)

  let graph_bytes dev attr =
    bytes "cuDeviceGetGraphMemAttribute" ((Lazy.force graph_attribute) dev attr)
end

let read () =
  let device = Driver.device () in
  let pool = Driver.pool device in
  let stream_bytes = Driver.pool_bytes pool and graph_bytes = Driver.graph_bytes device in
  (* CUmemPool_attribute: release threshold, then reserved and used (current/high).
     CUgraphMem_attribute: used then reserved (current/high). Driver ABI enum values. *)
  let threshold = stream_bytes 4 in
  let stream =
    {
      reserved = stream_bytes 5;
      reserved_high = stream_bytes 6;
      used = stream_bytes 7;
      used_high = stream_bytes 8;
    }
  in
  let graph =
    {
      reserved = graph_bytes 2;
      reserved_high = graph_bytes 3;
      used = graph_bytes 0;
      used_high = graph_bytes 1;
    }
  in
  let free, _ = Cu.Device.get_free_and_total_mem () in
  {
    is_default = Ctypes.ptr_compare pool (Driver.default device) = 0;
    threshold;
    stream;
    graph;
    free;
  }

let report ?baseline phase { is_default; threshold; stream; graph; free } =
  let delta = Option.value_map baseline ~default:0 ~f:(fun b -> free - b.free) in
  let bytes = Unsigned.UInt64.to_string in
  Stdio.eprintf
    "%s: free=%d B (change %+d B), current pool is default=%b, release threshold=%s B, reserved \
     and used current/high in B (not part of the golden)\n\
     %!"
    phase free delta is_default (bytes threshold);
  let print name { reserved; reserved_high; used; used_high } =
    Stdio.eprintf "  %s reserved=%s/%s used=%s/%s (not part of the golden)\n%!" name
      (bytes reserved) (bytes reserved_high) (bytes used) (bytes used_high)
  in
  print "stream pool" stream;
  print "graph allocator" graph

(* Sample the synchronous driver callback before the replacement launches and can reuse backing. *)
let after_first_exec_destroy run =
  let previous = !Cu.cuda_call_hook and observed = ref None in
  Cu.cuda_call_hook :=
    Some
      (fun ~message ~status ->
        Option.iter previous ~f:(fun hook -> hook ~message ~status);
        if
          String.equal message "cu_graph_exec_destroy"
          && Cu.is_success status && Option.is_none !observed
        then observed := Some (read ()));
  let result = Exn.protect ~f:run ~finally:(fun () -> Cu.cuda_call_hook := previous) in
  (result, !observed)

(* The device-wide timer allows the two streams' intervals to be compared on different SMs. Each
   launch copies its scratch tag out before the free, so accidental aliasing is visible. *)
let overlap_source =
  {|extern "C" __global__ void overlap(volatile unsigned long long *scratch,
                                      unsigned long long *result, int tag) {
      unsigned long long start, finish;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
      scratch[0] = tag;
      do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(finish));
      } while (finish - start < 10000000);
      result[0] = start;
      result[1] = finish;
      result[2] = scratch[0];
    }|}

let streams () =
  let module S = Cu.Stream in
  let device = Cu.Context.get_device () in
  let attributes = Cu.Device.get_attributes device in
  let arch = (attributes.compute_capability_major * 10) + attributes.compute_capability_minor in
  let ptx =
    Nvrtc.compile_to_ptx ~cu_src:overlap_source ~name:"cuda_pool_overlap.cu"
      ~options:[ Printf.sprintf "--gpu-architecture=compute_%d" arch ]
      ~with_debug:false
  in
  let kernel = Cu.Module.get_function (Cu.Module.load_data_ex ptx []) ~name:"overlap" in
  let a = S.create ~non_blocking:true () and b = S.create ~non_blocking:true () in
  let a_result = Cu.Deviceptr.mem_alloc ~size_in_bytes:24 in
  let b_result = Cu.Deviceptr.mem_alloc ~size_in_bytes:24 in
  let baseline = read () in
  report "before either stream allocation" baseline;
  (* This is the single-request size from the gh-ocannl-2263 witness. Both allocations are submitted
     before either free, so neither can reuse the other's allocation. *)
  let a_scratch = S.mem_alloc a ~size_in_bytes:66_304 in
  let b_scratch = S.mem_alloc b ~size_in_bytes:66_304 in
  report ~baseline "both scratch allocations submitted" (read ());
  let launch stream scratch result tag =
    S.launch_kernel kernel ~grid_dim_x:1 ~block_dim_x:1 ~shared_mem_bytes:0 stream
      [ S.Tensor scratch; S.Tensor result; S.Int tag ]
  in
  launch a a_scratch a_result 1;
  launch b b_scratch b_result 2;
  S.mem_free a a_scratch;
  S.mem_free b b_scratch;
  report ~baseline "both scratch frees submitted" (read ());
  S.synchronize a;
  S.synchronize b;
  report ~baseline "both streams synchronized" (read ());
  let get ptr =
    let result = Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout 3 in
    Cu.Deviceptr.memcpy_D_to_H ~dst:(Bigarray.genarray_of_array1 result) ~src:ptr ();
    Cu.Deviceptr.mem_free ptr;
    (result.{0}, result.{1}, result.{2})
  in
  let a_start, a_finish, a_tag = get a_result and b_start, b_finish, b_tag = get b_result in
  Verdict.p "both device kernel intervals overlapped"
    Int64.(max a_start b_start < min a_finish b_finish);
  Verdict.p "overlapping launches returned distinct scratch tags" Int64.(a_tag = 1L && b_tag = 2L)
